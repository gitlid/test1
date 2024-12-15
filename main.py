# encoding: utf-8


import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as trans
from Dataloader import DIODE_Dataset, Normalize, RandomChannel, ToTensor, Rescale
from models.HidingUNet import UnetGenerator
from models.DDNet import DDNet
from vgg import Vgg16
from loss import criterion
from metrics import DepthEstimateScore
import torch.cuda.amp as amp
import gc


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="test",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=16,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=2,
                    help='input batch size')
parser.add_argument('--imageSize', type=str, default=384,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=200,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default='data-x/LAPTOP-33576NBL__2024-12-14-20_54_01/checkPoints/Hnet_epoch_0,sumloss=0.970695,ssimloss=0.191964.pth',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Dnet', default='data-x/LAPTOP-33576NBL__2024-12-14-02_28_13/checkPoints/netD_epoch_0,sumloss=0.495543,mseloss=0.056215.pth',
                    help="path to Discriminator (to continue training)")
parser.add_argument('--trainpics', default='/data-x/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='/data-x/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='/data-x/',
                    help='folder to output test images')
parser.add_argument('--runfolder', default='/data-x/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='/data-x/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='/data-x/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='/data-x/',
                    help='folder to save the experiment codes')

parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default=True, help='test mode, you need give the test pics dirs in this param')
parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=200, help='the frequency of save the resultPic')


#datasets to train
parser.add_argument('--datasets', type=str, default='/home/roglnld/PycharmProjects/Deep_learning/DIODE_Dataset',)


#hyperparameter of loss

parser.add_argument('--beta', type=float, default=1,)
parser.add_argument('--betal1', type=float, default=1,)

parser.add_argument('--betamse', type=float, default=1,
                    help='hyper parameter of beta: mse_loss')
parser.add_argument('--betavgg', type=float, default=1,
                    help='hyper parameter of beta: vgg_loss')
parser.add_argument('--num_downs', type=int, default=7, help='nums of  Unet downsample')


def main():
    ############### define global parameters ###############
    global opt, optimizer, writer, logPath, scheduler, device
    global val_loader, smallestLoss,  mse_loss,\
        l1_loss, vgg, loss_fn, scaler

    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    elif opt.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cudnn.benchmark = True

    ############  create the dirs to save the result #############
    current_dictory = os.getcwd()
    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    experiment_dir = opt.hostname + "_" + opt.remark + "_" + cur_time
    opt.outckpts = current_dictory + opt.outckpts + experiment_dir + "/checkPoints"
    opt.trainpics = current_dictory + opt.trainpics + experiment_dir + "/trainPics"
    opt.validationpics = current_dictory + opt.validationpics + experiment_dir + "/validationPics"
    opt.outlogs = current_dictory + opt.outlogs + experiment_dir + "/trainingLogs"
    opt.outcodes = current_dictory + opt.outcodes + experiment_dir + "/codes"
    opt.testPics = current_dictory + opt.testPics + experiment_dir + "/testPics"
    opt.runfolder = current_dictory + opt.runfolder + experiment_dir + "/run"
    test = opt.test
    if not os.path.exists(opt.outckpts):
        os.makedirs(opt.outckpts)
    if not os.path.exists(opt.trainpics):
        os.makedirs(opt.trainpics)
    if not os.path.exists(opt.validationpics):
        os.makedirs(opt.validationpics)
    if not os.path.exists(opt.outlogs):
        os.makedirs(opt.outlogs)
    if not os.path.exists(opt.outcodes):
        os.makedirs(opt.outcodes)
    if not os.path.exists(opt.runfolder):
        os.makedirs(opt.runfolder)        
    if (not os.path.exists(opt.testPics)) and opt.test != '':
        os.makedirs(opt.testPics)

    # opt.imageSize = eval(opt.imageSize)



    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    print_log(str(opt), logPath)
    save_current_codes(opt.outcodes)
    # tensorboardX writer
    writer = SummaryWriter(log_dir=opt.runfolder, comment='**' + opt.hostname + "_" + opt.remark)

    
    DATA_DIR = opt.datasets
    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'val')

    # print((opt.imageSize,) * 2)
    train_dataset = DIODE_Dataset(
        traindir,  
        trans.Compose([
            Rescale((opt.imageSize,) * 2),
            RandomChannel(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                      max_depth=10)
        ]))
    val_dataset = DIODE_Dataset(
        valdir,  
        trans.Compose([
            Rescale((opt.imageSize,) * 2),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                      max_depth=10)
        ]))


    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                              shuffle=True, num_workers=int(opt.workers))

    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                            shuffle=False, num_workers=int(opt.workers))    	


    # Hnet = UnetGenerator(input_nc=3, output_nc=1, num_downs= opt.num_downs, output_function=nn.Sigmoid)
    # Hnet.to(device)
    # Hnet.apply(weights_init)
    # netName = 'Hnet'


    Dnet = DDNet()
    Dnet.to(device)
    Dnet.apply(weights_init)
    netName = 'Dnet'

    # setup optimizer
    # optimizer = optim.AdamW(Hnet.parameters(), lr=opt.lr, weight_decay=0.01)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lrs, epochs=self.config.epochs, steps_per_epoch=len(self.train_loader),
    #                                          cycle_momentum=self.config.cycle_momentum,
    #                                          base_momentum=0.85, max_momentum=0.95, div_factor=self.config.div_factor, final_div_factor=self.config.final_div_factor, pct_start=self.config.pct_start, three_phase=self.config.three_phase)

    # Hnet = torch.load(opt.Hnet)


    if not test:
        # optimizer2 = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.2, patience=5, verbose=True)
        optimizer = optim.Adam(Dnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)


    # if opt.Hnet != "":
    #     Hnet.load_state_dict(torch.load(opt.Hnet))
    # if opt.ngpu > 1:
    #     Hnet = torch.nn.DataParallel(Hnet).to(device)
    # print_network(Hnet)
    # net = Hnet
    # if opt.Rnet != '':
    #     Rnet.load_state_dict(torch.load(opt.Rnet))
    # if opt.ngpu > 1:
    #     Rnet = torch.nn.DataParallel(Rnet).to(device)
    # print_network(Rnet)
    # net = Rnet

    if opt.Dnet != '':
        Dnet.load_state_dict(torch.load(opt.Dnet))
    if opt.ngpu > 1:
        Dnet = torch.nn.DataParallel(Dnet).to(device)
    print_network(Dnet)
    net = Dnet


    # define loss
    mse_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)
    vgg = Vgg16(requires_grad=False).to(device)
    vgg.eval()
    loss_fn = criterion
    # Gradl1_loss = GradL1Loss()
    scaler = amp.GradScaler(enabled=False)

    if not test:
        smallestLoss = 10000
        # smallestLoss_2 = 10000
        print_log("training is beginning .......................................................", logPath)
        for epoch in range(opt.niter):
            ######################## train ##########################################
            train(train_loader, epoch, net=net)

            ####################### validation  #####################################
            val_sumloss, a1, abs_rel, val_loss = validation(val_loader, epoch, net=net)

            ####################### adjust learning rate ############################
            scheduler.step(val_loss)

            # save the best model parameters
            if abs_rel < globals()["smallestLoss"]:
                globals()["smallestLoss"] = abs_rel

                torch.save(net.state_dict(),
                           '%s/%s_epoch_%d,sumloss=%.6f,abs_rel_loss=%.6f.pth' % (
                               opt.outckpts, netName, epoch, val_sumloss, abs_rel))

            # if abs_rel_2 < globals()["smallestLoss_2"]:
            #     globals()["smallestLoss_2"] = abs_rel_2
            #     torch.save(net2.state_dict(),
            #                '%s/netR_epoch_%d,sumloss=%.6f,ssimloss=%.6f.pth' % (
            #                    opt.outckpts, epoch, val_sumloss_2, val_loss_2
            #                ))
            gc.collect()
    else:
        val_sumloss, a1, abs_rel, val_loss = validation(val_loader, 0, net=net)

        # print(f"val_mseloss: %.6f"
        #       f"")
    writer.close()


def train(train_loader, epoch, net):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # Losses_gradl1 = AverageMeter()
    # losses_mse = AverageMeter()
    # log_mse_losses = AverageMeter()
    # Vgglosses = AverageMeter()
    Vallosses = AverageMeter()
    # total_losses = AverageMeter()
    # Vallosses2 = AverageMeter()


    # switch to train mode
    net.train()
    # net2.train()

    start_time = time.time()

    for i, (data, img_name, mask) in enumerate(train_loader):
        data_time.update(time.time() - start_time)

        input_img, depth_gt = data['image'], data['depth']
        this_batch_size = int(input_img.size()[0])
        input_var = input_img.to(device)
        depth_gt_var = depth_gt.to(device)

        # Compute output
        pred_depth = net(input_var)
        # pred_depth2 = net2(input_var)

        # Calculate MSE loss
        # loss_mse = mse_loss(pred_depth, depth_gt_var) * opt.betamse
        # log_los_mse = mse_loss(torch.log(pred_depth + 1), torch.log(depth_gt_var + 1))

        # Calculate L1 loss
        # loss_l1 = l1_loss(pred_depth, depth_gt_var) * opt.betal1


        # container_depth = depth_gt_var.repeat(1, 3, 1, 1)
        # container_pred = pred_depth.repeat(1, 3, 1, 1)
        # vgg_loss = mse_loss(vgg(container_pred).relu2_2, vgg(container_depth).relu2_2) * opt.betavgg

        # val_loss = silog_loss(pred_depth, depth_gt_var, mask=mask) * opt.beta
        val_loss, pred_depth = loss_fn(depth_gt_var, pred_depth, mask, device=device)
        # val_loss2, pred_depth2 = loss_fn(depth_gt_var, pred_depth2, mask, device=device)

        # gradl1_loss = Gradl1_loss(pred_depth, depth_gt_var, mask,interpolate=False)
        val_loss *= opt.beta
        # val_loss2 *= opt.beta
        # val_loss += gradl1_loss * 0.3
        # Combine losses with appropriate weights
        # total_loss = val_loss

        # Record loss
        # losses_mse.update(loss_mse.item(), this_batch_size)
        # log_mse_losses.update(log_los_mse.item(), this_batch_size)
        # Losses_gradl1.update(gradl1_loss.item(), this_batch_size)
        # Vgglosses.update(vgg_loss.item(), this_batch_size)
        Vallosses.update(val_loss.item(), this_batch_size)
        # Vallosses2.update(val_loss2.item(), this_batch_size)
        # total_losses.update(total_loss.item(), this_batch_size)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(val_loss).backward()
        # total_loss.backward()
        scaler.step(optimizer)

        # optimizer2.zero_grad()
        # scaler.scale(val_loss2).backward()
        # total_loss.backward()
        # scaler.step(optimizer2)



        scaler.update()
        # Measure elapsed time
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = f'''Epoch: [{epoch}][{i}/{len(train_loader)}]\t
                          Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t
                          Data {data_time.val:.3f} ({data_time.avg:.3f})\t
                          Loss {Vallosses.val:.4f} ({Vallosses.avg:.4f})\t'''
                          # MSE Loss {losses_mse.val:.4f} ({losses_mse.avg:.4f})\t
                          # Log MSE Loss {log_mse_losses.val:.4f} ({log_mse_losses.avg:.4f})\t
                          # VGG Loss {Vgglosses.val:.4f} ({Vgglosses.avg:.4f})\t
                          # L1 Loss {losses_l1.val:.4f} ({losses_l1.avg:.4f})\t
                          # Total Loss {total_losses.val:.4f} ({total_losses.avg:.4f})
                          # Loss2 {Vallosses2.val:.4f} ({Vallosses2.avg:.4f})\t


        if i % opt.logFrequency == 0:
            print_log(log, logPath)
        else:
            print_log(log, logPath, False)

        if epoch % 1 == 0 and i % opt.resultPicFrequency == 0:
            diff = (pred_depth.detach() - depth_gt_var).abs()
            save_result_pic(this_batch_size, input_var.cpu(), depth_gt_var.cpu(), pred_depth.detach().cpu(), diff.detach().cpu(), epoch, i, opt.trainpics)

    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + f"epoch learning rate: optimizer_lr = {optimizer.param_groups[0]['lr']:.8f}\n"
    epoch_log = epoch_log + (f"epoch_loss={Vallosses.avg:.6f}\t"
                             # f"epoch_loss2={Vallosses2.avg:.6f}\t"
                             # f"epoch_Losses_grad_l1={Losses_gradl1.avg:.6f}"
                             )
                             # f"epoch_loss_mse={losses_mse.avg:.6f}\t"
                             # f"epoch_loss_log_mse={log_mse_losses.avg:.6f}\n"
                              # f"epoch_loss_vgg={Vgglosses.avg:.6f}\n"
                             # f"epoch_total_losses={total_losses.avg:.6f}")


    print_log(epoch_log, logPath)


    writer.add_scalar("lr/lr_", optimizer.param_groups[0]['lr'], epoch)
    # writer.add_scalar("lr/lr_2", optimizer2.param_groups[0]['lr'], epoch)
    # writer.add_scalar("lr/beta", opt.beta, epoch)
    # writer.add_scalar("loss/Losses_grad_l1", Losses_gradl1.avg, epoch)
    # writer.add_scalar("loss/losses_mse", losses_mse.avg, epoch)
    # writer.add_scalar("loss/log_mse_loss", log_mse_losses.avg, epoch)
    # writer.add_scalar("loss/vgg_loss", Vgglosses.avg, epoch)
    writer.add_scalar("loss/loss", Vallosses.avg, epoch)
    # writer.add_scalar("loss/loss2", Vallosses2.avg, epoch)
    # writer.add_scalar("loss/total_loss", total_losses.avg, epoch)


def validation(val_loader,  epoch, net):
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    net.eval()
    # net2.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_l1 = AverageMeter()
    losses_mse = AverageMeter()
    # abs_rel_diff = AverageMeter()
    # log_mse_losses = AverageMeter()
    Vallosses = AverageMeter()
    Vgglosses = AverageMeter()
    total_losses = AverageMeter()
    # a1_thresh = AverageMeter()
    # a2_thresh = AverageMeter()
    # a3_thresh = AverageMeter()
    scores = DepthEstimateScore()


    # losses_l1_2 = AverageMeter()
    # losses_mse_2 = AverageMeter()
    # abs_rel_diff_2 = AverageMeter()
    # # log_mse_losses = AverageMeter()
    # Vallosses_2 = AverageMeter()
    # Vgglosses_2 = AverageMeter()
    # total_losses_2 = AverageMeter()
    # a1_thresh_2 = AverageMeter()
    # a2_thresh_2 = AverageMeter()
    # a3_thresh_2 = AverageMeter()


    # Tensor type
    # Tensor = torch.cuda.FloatTensor
    with torch.no_grad():

        for i, (data, img_name, mask) in enumerate(val_loader):
            data_time.update(time.time() - start_time)

            input_img, depth_gt = data['image'], data['depth']
            this_batch_size = int(input_img.size()[0])
            input_var = input_img.to(device)
            depth_gt_var = depth_gt.to(device)
            # Compute output
            pred_depth = net(input_var)
            # pred_depth2 = net2(input_var)

            # Calculate MSE loss
            val_loss, pred_depth = loss_fn(depth_gt_var, pred_depth, mask, device=device)
            val_loss *= opt.beta

            # val_loss_2, pred_depth2 = loss_fn(depth_gt_var, pred_depth2, mask, device=device)
            # val_loss_2 *= opt.beta

            loss_mse = mse_loss(pred_depth, depth_gt_var) * opt.betamse

            # loss_mse_2 = mse_loss(pred_depth2, depth_gt_var) * opt.betamse

            # log_los_mse = mse_loss(torch.log(pred_depth + 1), torch.log(depth_gt_var + 1))
            # abs_rel = torch.mean(torch.abs(depth_gt_var[mask[:,None]] - pred_depth[mask[:,None]]) / depth_gt_var[mask[:,None]])

            # abs_rel_2 = torch.mean(torch.abs(depth_gt_var[mask[:,None]] - pred_depth2[mask[:,None]]) / depth_gt_var[mask[:,None]])

            # Calculate L1 loss
            loss_l1 = l1_loss(pred_depth, depth_gt_var) * opt.betal1
            # loss_l1_2 = l1_loss(pred_depth2, depth_gt_var) * opt.betal1



            # container_depth = depth_gt_var.repeat(1, 3, 1, 1)
            # container_pred = pred_de.repeat(1, 3, 1, 1)
            vgg_loss = mse_loss(vgg(pred_depth.repeat(1, 3, 1, 1)).relu2_2, vgg(depth_gt_var.repeat(1, 3, 1, 1)).relu2_2) * opt.betavgg

            # vgg_loss_2  = mse_loss(vgg(pred_depth2.repeat(1, 3, 1, 1)).relu2_2, vgg(depth_gt_var.repeat(1, 3, 1, 1)).relu2_2) * opt.betavgg
            # val_loss = silog_loss(pred_depth, depth_gt_var, mask=mask) * opt.beta

            # Combine losses with appropriate weights
            total_loss =  loss_mse + loss_l1 + vgg_loss + val_loss

            # total_loss_2 = loss_mse_2 + loss_l1_2 + vgg_loss_2 + val_loss_2 + abs_rel_2

            # a1, a2, a3 = compute_errors(pred_depth.detach(), depth_gt_var.detach())

            # a1_2, a2_2, a3_2 = compute_errors(pred_depth2.detach(), depth_gt_var.detach())


            # Record loss
            losses_mse.update(loss_mse.item(), this_batch_size)
            # log_mse_losses.update(log_los_mse.item(), this_batch_size)
            # abs_rel_diff.update(abs_rel, this_batch_size)
            losses_l1.update(loss_l1.item(), this_batch_size)
            Vgglosses.update(vgg_loss.item(), this_batch_size)
            Vallosses.update(val_loss.item(), this_batch_size)
            total_losses.update(total_loss.item(), this_batch_size)
            # a1_thresh.update(a1, this_batch_size)
            # a2_thresh.update(a2, this_batch_size)
            # a3_thresh.update(a3, this_batch_size)
            scores.update(label_true=depth_gt_var[mask[:,None]].detach().cpu(), label_pred=pred_depth[mask[:,None]].detach().cpu())

            # losses_mse_2.update(loss_mse_2.item(), this_batch_size)
            # log_mse_losses.update(log_los_mse.item(), this_batch_size)
            # abs_rel_diff_2.update(abs_rel_2, this_batch_size)
            # losses_l1_2.update(loss_l1_2.item(), this_batch_size)
            # Vgglosses_2.update(vgg_loss_2.item(), this_batch_size)
            # Vallosses_2.update(val_loss_2.item(), this_batch_size)
            # total_losses_2.update(total_loss_2.item(), this_batch_size)
            # a1_thresh_2.update(a1_2, this_batch_size)
            # a2_thresh_2.update(a2_2, this_batch_size)
            # a3_thresh_2.update(a3_2, this_batch_size)


            # Measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()


            if i % 50 == 0:
                diff = (pred_depth.detach() - depth_gt_var).abs()
                save_result_pic(this_batch_size, input_var.cpu(), depth_gt_var.cpu(), pred_depth.detach().cpu(), diff.detach().cpu(), epoch, i, opt.validationpics)

        val_log = "validation[%d] time is %.4f======================================================================" % (
            epoch, batch_time.sum) + "\n"
        # val_log = val_log + 'net1======================================================================\n'
        val_log = val_log + (f"total_losses={total_losses.avg:.6f}\n"
                             f"losses_l1={losses_l1.avg:.6f}\tloss_mse={losses_mse.avg:.6f}\tloss_vgg={Vgglosses.avg:.6f}\n"
                             f"loss_val={Vallosses.avg:.6f}")
        # val_log = val_log + f"\n a1={a1_thresh.avg:.6f}\t a2={a2_thresh.avg:.6f}\t a3={a3_thresh.avg:.6f}"
        print_log(val_log, logPath)
        score = scores.get_scores()
        for k, v in score.items():
            print_log("{}: {}".format(k, v), logPath)
            # logger.info("{}: {}".format(k, v))
            writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

        # val_log = 'net2======================================================================\n'
        # val_log_2 = val_log + (f"losses_l1={losses_l1_2.avg:.6f}\tloss_mse={losses_mse_2.avg:.6f}\tloss_vgg={Vgglosses_2.avg:.6f}\n"
        #                      f"loss_val={Vallosses_2.avg:.6f}\tabs_rel={abs_rel_diff_2.avg:.6f}\n"
        #                      f"total_losses={total_losses_2.avg:.6f}")
        #
        # print_log(val_log_2, logPath)

        # writer.add_scalar("lr/_lr", optimizer.param_groups[0]['lr'], epoch)
        # writer.add_scalar("lr/beta", opt.beta, epoch)
        writer.add_scalar("val/loss/losses_l1", losses_l1.avg, epoch)
        writer.add_scalar("val/loss/losses_mse", losses_mse.avg, epoch)
        # writer.add_scalar("val/loss/abs_rel", abs_rel_diff.avg, epoch)
        writer.add_scalar("val/loss/vgg_loss", Vgglosses.avg, epoch)
        writer.add_scalar("val/loss/val_loss", Vallosses.avg, epoch)
        writer.add_scalar("val/loss/total_loss", total_losses.avg, epoch)
        # writer.add_scalar("val_metrics/a1", a1_thresh.avg, epoch)
        # writer.add_scalar("val_metrics/a2", a2_thresh.avg, epoch)
        # writer.add_scalar("val_metrics/a3", a3_thresh.avg, epoch)

        # writer.add_scalar("val/loss/losses_l1_2", losses_l1_2.avg, epoch)
        # writer.add_scalar("val/loss/losses_mse_2", losses_mse_2.avg, epoch)
        # writer.add_scalar("val/loss/abs_rel_2", abs_rel_diff_2.avg, epoch)
        # writer.add_scalar("val/loss/vgg_loss_2", Vgglosses_2.avg, epoch)
        # writer.add_scalar("val/loss/val_loss_2", Vallosses_2.avg, epoch)
        # writer.add_scalar("val/loss/total_loss_2", total_losses_2.avg, epoch)
        # writer.add_scalar("val_metrics/a1_2", a1_thresh_2.avg, epoch)
        # writer.add_scalar("val_metrics/a2_2", a2_thresh_2.avg, epoch)
        # writer.add_scalar("val_metrics/a3_2", a3_thresh_2.avg, epoch)
    print("#################################################### validation end ########################################################")

    return total_losses.avg, score['a1'], score['abs_rel'], Vallosses.avg


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)  
    cur_work_dir, mainfile = os.path.split(main_file_path)

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)

    loss_file = cur_work_dir + "/loss.py"
    new_loss_file_path = des_path + "/loss.py"
    shutil.copyfile(loss_file, new_loss_file_path)

    dateload_file = cur_work_dir + "/Dataloader.py"
    new_dateload_file_path = des_path + "/Dataloader.py"
    shutil.copyfile(dateload_file, new_dateload_file_path)


# print the training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log into files
    if not opt.debug:
        # write the log into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


# save result pic and the coverImg filePath and the secretImg filePath
def save_result_pic(this_batch_size, input_img, depth_gt, pred_depth, diff, epoch, i, save_path):
    mean=torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std=torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    dim = (1,2,3)

    input_img = input_img * std + mean
    input_img = F.interpolate(input_img, [192, 256], mode='bilinear', align_corners=False)
    # input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
    # depth_gt = depth_gt.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
    # depth_gt = (depth_gt - depth_gt.min()) / (depth_gt.max() - depth_gt.min())
    min_depth, max_depth = depth_gt.amin(dim=dim, keepdim=True), depth_gt.amax(dim=dim, keepdim=True)
    depth_gt = (depth_gt - min_depth) / (max_depth - min_depth)
    depth_gt = F.interpolate(depth_gt, [192, 256], mode='bilinear', align_corners=False)
    depth_gt = depth_gt.repeat(1, 3, 1, 1)
    # pred_depth = pred_depth.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
    min_depth, max_depth = pred_depth.amin(dim=dim, keepdim=True), pred_depth.amax(dim=dim, keepdim=True)
    pred_depth = (pred_depth - min_depth) / (max_depth - min_depth)
    pred_depth = F.interpolate(pred_depth, [192, 256], mode='bilinear', align_corners=False)
    pred_depth = pred_depth.repeat(1, 3, 1, 1)
    # diff = diff.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
    min_depth, max_depth = diff.amin(dim=dim, keepdim=True), diff.amax(dim=dim, keepdim=True)
    diff = (diff - min_depth) / (max_depth - min_depth)
    diff = F.interpolate(diff, [192, 256], mode='bilinear', align_corners=False)
    diff = diff.repeat(1, 3, 1, 1)


    showResult = torch.cat([input_img, depth_gt, pred_depth, diff, ], 0)

    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    vutils.save_image(showResult, resultImgName, nrow=this_batch_size, padding=1, normalize=False)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_errors(pred, gt):
    tresh = torch.maximum(gt / pred, pred / gt)

    a1 = (tresh < 1.25).float().mean()
    a2 = (tresh < 1.25 ** 2).float().mean()
    a3 = (tresh < 1.25 ** 3).float().mean()

    return a1, a2, a3




if __name__ == '__main__':
    main()
