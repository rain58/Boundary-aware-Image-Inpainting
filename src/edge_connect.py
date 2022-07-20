import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import torch
# import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel, InpaintingModel, InpaintingDEModel,DepthModel, GatedDEInpaintingModel
import matplotlib.pyplot as plt
from .utils import Progbar, create_dir, stitch_images, imsave,log_stage2
from PIL import Image
import torch
import time
from skimage.feature import canny
from .metrics import PSNR, EdgeAccuracy,DepthAccuracy
import tensorboardX as tbx
# from torch.utils.tensorboard import SummaryWriter

class EdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'
        elif config.MODEL == 5:
            model_name = 'depth'
        elif config.MODEL == 6:
            model_name = 'gated_edge_depth_inpaint'
        elif config.MODEL == 7:
            model_name = 'edge_depth_inpaint'
        elif config.MODEL == 10:
            model_name = 'depth_inpaint'

        
        self.path_c = config.PATH
        self.debug = False
        self.sigma = config.SIGMA
        self.comment = config.COMMENT
        self.dedge = config.DEDGE
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.depth_model = DepthModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.gated_inpaint_model = GatedDEInpaintingModel(config).to(config.DEVICE)
        self.inpaint_de_model = InpaintingDEModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)
        self.depthacc = DepthAccuracy(config.DEPTH_THRESHOLD).to(config.DEVICE)
        

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, config.TEST_DEPTH_FLIST,  augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, config.TRAIN_DEPTH_FLIST,  augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, config.VAL_DEPTH_FLIST,  augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')

        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + self.comment + '.dat')
        self.writer = tbx.SummaryWriter(log_dir="{}/logs/{}_{}".format(self.path_c,model_name,self.comment))

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        elif self.config.MODEL == 5:
            self.depth_model.load()

        elif self.config.MODEL == 6:
            self.depth_model.load()
            self.edge_model.load()
            self.gated_inpaint_model.load()

        elif self.config.MODEL == 7:
            self.depth_model.load()
            self.edge_model.load()
            self.inpaint_de_model.load()

        elif self.config.MODEL == 10:
            self.depth_model.load()
            self.inpaint_model.load()

        else:
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()

        elif self.config.MODEL == 5:
            self.depth_model.save()

        elif self.config.MODEL == 6:
            self.gated_inpaint_model.save()

        elif self.config.MODEL == 7:
            self.inpaint_de_model.save()
        
        elif self.config.MODEL == 10:
            self.inpaint_model.save()

        else:
            self.edge_model.save()
            self.inpaint_model.save()

    def save_name(self):
        if self.config.MODEL == 1:
            self.edge_model.save_name()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save_name()

        elif self.config.MODEL == 5:
            self.depth_model.save_name()

        elif self.config.MODEL == 6:
            self.gated_inpaint_model.save_name()

        elif self.config.MODEL == 7:
            self.inpaint_de_model.save_name()

        elif self.config.MODEL == 10:
            self.inpaint_model.save_name()

        else:
            self.edge_model.save_name()
            self.inpaint_model.save_name()

    
    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=12,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        print("MAX iteration = ",max_iteration)
        total = len(self.train_dataset)
        print("DATASET Total = ",total)
        loss_dict = {"l_d1":0,"l_g1":0,"l_fm":0,"precision":0,"recall":0,"l_d2":0,"l_g2":0,"l_l1":0,"l_per":0,"l_sty":0,"psnr":0,"mae":0,"mse":0}

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.edge_model.train()
                self.depth_model.train()
                self.inpaint_model.train()
                self.gated_inpaint_model.train()
                self.inpaint_de_model.train()
                images, images_gray, edges, masks, depths = self.cuda(*items)
        
                # edge model
                if model == 1:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks,eval=False)
                    # metrics
                    precision, recall = self.edgeacc(edges * masks, outputs * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))
                    # backward
                    self.edge_model.backward(gen_loss, dis_loss)
                    iteration = self.edge_model.iteration
                    loss_dict = log_stage2(loss_dict,logs)

                # inpaint model
                elif model == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))
                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration
                    loss_dict = log_stage2(loss_dict,logs)

                # inpaint with edge model
                elif model == 3:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.edge_model(images_gray, edges, masks)
                        outputs = outputs * masks + edges * (1 - masks)
                    else:
                        outputs = edges

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks, eval=False)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))
                    
                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration
                    loss_dict = log_stage2(loss_dict,logs)                    

                elif model == 5:
                    if self.dedge == 1:
                        depths = (depths*255).to(torch.device("cpu")).detach().numpy().copy()
                        for i in range(depths.shape[0]):
                            depths[i,0,:] = canny(depths[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                            
                        depths = torch.from_numpy(depths.astype(np.float32)).clone()
                        depths = depths.to(torch.device("cuda"))
                    # train
                    outputs, gen_loss, dis_loss, logs = self.depth_model.process(images_gray, depths, masks,eval=False)
                    # metrics
                    mse,mae = self.depthacc(depths * masks, outputs * masks)
                    logs.append(('mse', mse.item()))
                    logs.append(('mae', mae.item()))
                    # logs.append(('recall', recall.item()))

                    # backward
                    self.depth_model.backward(gen_loss, dis_loss)
                    iteration = self.depth_model.iteration
                    loss_dict = log_stage2(loss_dict,logs)

                elif model == 6:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs_edge = self.edge_model(images_gray, edges, masks)
                        outputs_edge = outputs_edge * masks + edges * (1 - masks)
                        outputs_depth = self.depth_model(images_gray, depths, masks)
                        outputs_depth = outputs_depth * masks + depths * (1 - masks)
                    else:
                        outputs_edge = edges
                        outputs_depth = depths
                    outputs_depth = outputs_depth/ torch.max(outputs_depth)

                    if self.dedge == 1:
                        outputs_depth = (outputs_depth*255).to(torch.device("cpu")).detach().numpy().copy()
                        for i in range(outputs_depth.shape[0]):
                            outputs_depth[i,0,:] = canny(outputs_depth[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                            
                        outputs_depth = torch.from_numpy(outputs_depth.astype(np.float32)).clone()
                        outputs_depth = outputs_depth.to(torch.device("cuda"))
                    
                    outputs, gen_loss, dis_loss, logs = self.gated_inpaint_model.process(images, outputs_edge.detach(), outputs_depth.detach(), masks,eval=False)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.gated_inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.gated_inpaint_model.iteration
                    loss_dict = log_stage2(loss_dict,logs)
                    

                elif model == 7:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs_edge = self.edge_model(images_gray, edges, masks)
                        outputs_edge = outputs_edge * masks + edges * (1 - masks)
                        outputs_depth = self.depth_model(images_gray, depths, masks)
                        outputs_depth = outputs_depth * masks + depths * (1 - masks)
                    else:
                        outputs_edge = edges
                        outputs_depth = depths
                    
                    if self.dedge == 1:
                        #dedge
                        outputs_depth = (outputs_depth*255).to(torch.device("cpu")).detach().numpy().copy()
                        
                        for i in range(outputs_depth.shape[0]):
                            outputs_depth[i,0,:] = canny(outputs_depth[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                            
                        outputs_depth = torch.from_numpy(outputs_depth.astype(np.float32)).clone()
                        outputs_depth = outputs_depth.to(torch.device("cuda"))

                    outputs, gen_loss, dis_loss, logs = self.inpaint_de_model.process(images, outputs_edge.detach(),outputs_depth.detach(), masks, eval=False)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))
                    # backward
                    self.inpaint_de_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_de_model.iteration
                    loss_dict = log_stage2(loss_dict,logs)

                # inpaint with edge model
                elif model == 10:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.depth_model(images_gray, depths, masks)
                        outputs = outputs * masks + depths * (1 - masks)
                    else:
                        outputs = depths

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks, eval=False)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))
                    
                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration
                    loss_dict = log_stage2(loss_dict,logs)

                # joint model
                else:
                    # train
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks,eval=False)
                    e_outputs = e_outputs * masks + edges * (1 - masks)
                    i_outputs, i_gen_loss, i_dis_loss, logs = self.inpaint_model.process(images, e_outputs, masks,eval=False)
                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(i_gen_loss, i_dis_loss,retain_graph=True)
                    self.edge_model.backward(e_gen_loss, e_dis_loss)
                    iteration = self.inpaint_model.iteration
                    loss_dict = log_stage2(loss_dict,logs)
                    
                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

                if iteration % (self.config.SAVE_INTERVAL*2) == 0:
                    self.save_name()

                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:

                    self.log(logs)

                    loss_list = []
                    print(loss_dict)
                    for name,value in loss_dict.items():
                        if value != 0:
                            mean = value/self.config.LOG_INTERVAL
                            loss_list.append((name,mean))
                            loss_dict[name] = 0
                    for i in loss_list:
                       self.writer.add_scalar(i[0],i[1],int(iteration))

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

        self.writer.close()
        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )
        it = len(val_loader)
        model = self.config.MODEL
        total = len(self.val_dataset)
        eval_loss_dict = {"l_d1":0,"l_g1":0,"l_fm":0,"precision":0,"recall":0,"l_d2":0,"l_g2":0,"l_l1":0,"l_per":0,"l_sty":0,"psnr":0,"mae":0,"mse":0}


        self.edge_model.eval()
        self.depth_model.eval()
        self.inpaint_model.eval()
        self.gated_inpaint_model.eval()
        self.inpaint_de_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])

        for items in val_loader:
            images, images_gray, edges, masks,depths = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks,eval=True)

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))
                eval_loss_dict = log_stage2(eval_loss_dict,logs)

            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks,eval=True)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                eval_loss_dict = log_stage2(eval_loss_dict,logs)

            # inpaint with edge model
            elif model == 3:
                # eval
                outputs = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks,eval=True)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                eval_loss_dict = log_stage2(eval_loss_dict,logs)

            elif model == 5:
                # eval
                if self.dedge == 1:
                    depths = (depths*255).to(torch.device("cpu")).detach().numpy().copy()
                    for i in range(depths.shape[0]):
                        depths[i,0,:] = canny(depths[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                        
                    depths = torch.from_numpy(depths.astype(np.float32)).clone()
                    depths = depths.to(torch.device("cuda"))
                outputs, gen_loss, dis_loss, logs = self.depth_model.process(images_gray, depths, masks,eval=True)

                # metrics
                mse, mae = self.depthacc(depths * masks, depths * masks)
                logs.append(('mse', mse.item()))
                logs.append(('mae', mae.item()))
                eval_loss_dict = log_stage2(eval_loss_dict,logs)

            elif model == 6:
                # eval
                outputs_edge = self.edge_model(images_gray, edges, masks)
                outputs_edge = outputs_edge * masks + edges * (1 - masks)
                outputs_depth = self.depth_model(images_gray, depths, masks)
                outputs_depth = outputs_depth * masks + depths * (1 - masks)
                outputs_depth = outputs_depth / torch.max(outputs_depth)

                if self.dedge == 1:
                    outputs_depth = (outputs_depth*255).to(torch.device("cpu")).detach().numpy().copy()
                    for i in range(outputs_depth.shape[0]):
                        outputs_depth[i,0,:] = canny(outputs_depth[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                    outputs_depth = torch.from_numpy(outputs_depth.astype(np.float32)).clone()
                    outputs_depth = outputs_depth.to(torch.device("cuda"))
                

                outputs, gen_loss, dis_loss, logs = self.gated_inpaint_model.process(images, outputs_edge.detach(), outputs_depth.detach(), masks,eval=True)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                eval_loss_dict = log_stage2(eval_loss_dict,logs)

            elif model == 7:
                # eval
                outputs_edge = self.edge_model(images_gray, edges, masks)
                outputs_edge = outputs_edge * masks + edges * (1 - masks)
                outputs_depth = self.depth_model(images_gray, depths, masks)
                outputs_depth = outputs_depth * masks + depths * (1 - masks)

                if self.dedge == 1:
                    outputs_depth = (outputs_depth*255).to(torch.device("cpu")).detach().numpy().copy()
                    for i in range(outputs_depth.shape[0]):
                        outputs_depth[i,0,:] = canny(outputs_depth[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                    outputs_depth = torch.from_numpy(outputs_depth.astype(np.float32)).clone()
                    outputs_depth = outputs_depth.to(torch.device("cuda"))

                outputs, gen_loss, dis_loss, logs = self.inpaint_de_model.process(images, outputs_edge.detach(),outputs_depth.detach(), masks,eval=True)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                eval_loss_dict = log_stage2(eval_loss_dict,logs)

            # inpaint with edge model
            elif model == 10:
                # eval
                outputs = self.depth_model(images_gray, depths, masks)
                outputs = outputs * masks + edges * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks,eval=True)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                eval_loss_dict = log_stage2(eval_loss_dict,logs)

            # joint model
            else:
                # eval
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks,eval=True)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, logs = self.inpaint_model.process(images, e_outputs, masks,eval=True)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                e_logs.append(('pre', precision.item()))
                e_logs.append(('rec', recall.item()))
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                eval_loss_dict = log_stage2(eval_loss_dict,logs)

            logs = [("it", it), ] + logs
            progbar.add(len(images), values=logs)
            
        loss_list = []
        for name,value in eval_loss_dict.items():
            if value != 0:
                mean = value/it
                loss_list.append((name,mean))
                eval_loss_dict[name] = 0
        for i in loss_list:
            self.writer.add_scalar(i[0],i[1],self.gated_inpaint_model.iteration)

        if self.model_name == "gated_edge_depth_inpaint":
            for i in range(len(self.yy)):
                self.writer.add_scalar(self.yy[i][0],self.yy[i][1],self.gated_inpaint_model.iteration)
                
        elif self.model_name == "edge_depth_inpaint":
            for i in range(len(self.yy)):
                self.writer.add_scalar(self.yy[i][0],self.yy[i][1],self.inpaint_de_model.iteration)

        else:
            for i in range(len(self.yy)):
                eval_name = "eval_"+self.yy[i][0]
                self.writer.add_scalar(eval_name,self.yy[i][1],self.inpaint_model.iteration)
        print('\nEnd eval....')


    def test(self):
        self.edge_model.eval()
        self.depth_model.eval()
        self.inpaint_model.eval()
        self.gated_inpaint_model.eval()
        self.inpaint_de_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name, mask_name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks, depths= self.cuda(*items)

            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                precision, recall = self.edgeacc(edges * masks, outputs * masks)                
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                l1 = self.psnr(images,outputs_merged)
                l2 = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

            elif model == 5:
                if self.dedge == 1:
                    depths = (depths*255).to(torch.device("cpu")).detach().numpy().copy()
                    for i in range(depths.shape[0]):
                        depths[i,0,:] = canny(depths[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                        
                    depths = torch.from_numpy(depths.astype(np.float32)).clone()
                    depths = depths.to(torch.device("cuda"))
                outputs = self.depth_model(images_gray, depths, masks)
                mse,mae = self.depthacc(depths * masks, outputs * masks)
                outputs_merged = (outputs * masks) + (depths * (1 - masks))

                # Saving colormapped depth image
                ttt = outputs_merged.squeeze().cpu().detach().numpy()
                vmax = np.percentile(ttt, 95)
                normalizer = matplotlib.colors.Normalize(vmin=ttt.min(), vmax=vmax)
                mapper = matplotlib.cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(ttt)[:, :, :3] * 255).astype(np.uint8)
                outputs_merged = Image.fromarray(colormapped_im)

            elif model == 6:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs_depth = self.depth_model(images_gray, depths, masks).detach()
                outputs_depth = outputs_depth / torch.max(outputs_depth)
                if self.dedge == 1:
                        outputs_depth = (outputs_depth*255).to(torch.device("cpu")).detach().numpy().copy()
                        for i in range(outputs_depth.shape[0]):
                            outputs_depth[i,0,:] = canny(outputs_depth[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                        outputs_depth = torch.from_numpy(outputs_depth.astype(np.float32)).clone()
                        outputs_depth = outputs_depth.to(torch.device("cuda"))
                outputs = self.gated_inpaint_model(images, edges, outputs_depth.detach(), masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            elif model == 7:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs_depth = self.depth_model(images_gray, depths, masks).detach()
                if self.dedge == 1:
                        outputs_depth = (outputs_depth*255).to(torch.device("cpu")).detach().numpy().copy()
                        for i in range(outputs_depth.shape[0]):
                            outputs_depth[i,0,:] = canny(outputs_depth[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                        outputs_depth = torch.from_numpy(outputs_depth.astype(np.float32)).clone()
                        outputs_depth = outputs_depth.to(torch.device("cuda"))

                outputs = self.inpaint_de_model(images, edges, outputs_depth, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            elif model == 10:
                depths = self.depth_model(images_gray, depths, masks).detach()
                outputs = self.inpaint_model(images, depths, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # inpaint with edge model / joint model
            else:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            output = self.postprocess(outputs_merged)[0]
            output = torch.clamp(output,0,255)
            path = os.path.join(self.results_path, name)
            imsave(output, path)

            if self.debug:
                edges = self.postprocess(edges)[0]
                # depth = self.postprocess(outputs_depth)[0]
                # ttt = depth.squeeze().cpu().detach().numpy()
                # vmax = np.percentile(ttt, 95)
                # normalizer = matplotlib.colors.Normalize(vmin=ttt.min(), vmax=vmax)
                # mapper = matplotlib.cm.ScalarMappable(norm=normalizer, cmap='magma')
                # colormapped_im = (mapper.to_rgba(ttt)[:, :, :3] * 255).astype(np.uint8)
                # outputs_depth = Image.fromarray(colormapped_im)
                
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))
        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()
        self.depth_model.eval()
        self.inpaint_model.eval()
        self.gated_inpaint_model.eval()
        self.inpaint_de_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks, depths = self.cuda(*items)

        # edge model
        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks)
            outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        elif model == 5:
            iteration = self.depth_model.iteration
            inputs = (images_gray * (1 - masks)) + masks
            if self.dedge == 1:
                depths = (depths*255).to(torch.device("cpu")).detach().numpy().copy()
                for i in range(depths.shape[0]):
                    depths[i,0,:] = canny(depths[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                    
                depths = torch.from_numpy(depths.astype(np.float32)).clone()
                depths = depths.to(torch.device("cuda"))

            edges = depths
            outputs = self.depth_model(images_gray, depths, masks)
            outputs_merged = (outputs * masks) + (depths * (1 - masks))

        elif model == 6:
            iteration = self.gated_inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs_edge = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs_edge * masks + edges * (1 - masks)).detach()
            outputs_depth = self.depth_model(images_gray, depths, masks).detach()
            outputs_depth = (outputs_depth * masks + depths * (1 - masks)).detach()
            outputs_depth = outputs_depth/ torch.max(outputs_depth)

            if self.dedge == 1:
                outputs_depth = (outputs_depth*255).to(torch.device("cpu")).detach().numpy().copy()
                for i in range(outputs_depth.shape[0]):
                    outputs_depth[i,0,:] = canny(outputs_depth[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                outputs_depth = torch.from_numpy(outputs_depth.astype(np.float32)).clone()
                outputs_depth = outputs_depth.to(torch.device("cuda"))
            
            outputs = self.gated_inpaint_model(images, edges, outputs_depth, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        elif model == 7:
            iteration = self.inpaint_de_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs_edge = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs_edge * masks + edges * (1 - masks)).detach()
            outputs_depth = self.depth_model(images_gray, depths, masks).detach()
            outputs_depth = (outputs_depth * masks + depths * (1 - masks)).detach()
            if self.dedge == 1:
                outputs_depth = (outputs_depth*255).to(torch.device("cpu")).detach().numpy().copy()
                for i in range(outputs_depth.shape[0]):
                    outputs_depth[i,0,:] = canny(outputs_depth[i,0,:], sigma=2,low_threshold=7,high_threshold=20).astype(np.float)
                outputs_depth = torch.from_numpy(outputs_depth.astype(np.float32)).clone()
                outputs_depth = outputs_depth.to(torch.device("cuda"))

            outputs = self.inpaint_de_model(images, edges, outputs_depth, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))
        
        elif model == 10:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.depth_model(images_gray, depths, masks).detach()
            depths = (outputs * masks + depths * (1 - masks)).detach()
            outputs = self.inpaint_model(images, depths, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # inpaint with edge model / joint model
        else:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        if model == 6:
            images = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(edges),
                self.postprocess(outputs_depth),
                self.postprocess(outputs),
                self.postprocess(outputs_merged),
                img_per_row = image_per_row
            )

        elif model == 7:
            images = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(edges),
                self.postprocess(outputs_depth),
                self.postprocess(outputs),
                self.postprocess(outputs_merged),
                img_per_row = image_per_row
            )
        
        else:
            images = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(edges),
                self.postprocess(outputs),
                self.postprocess(outputs_merged),
                img_per_row = image_per_row
            )


        path = os.path.join(self.samples_path, self.model_name, self.comment)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0

        # #-1,1Normalization
        # img = (img +1)*127.5

        img = img.permute(0, 2, 3, 1)
        return img.int()
