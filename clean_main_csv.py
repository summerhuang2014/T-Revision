import os
import torch
import tools
import numpy as np
import data_load
import argparse, sys
import Lenet, Resnet
import torch.nn as nn
import torch.optim as optim
from loss import reweight_loss, reweighting_revision_loss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from transformer import transform_train, transform_test,transform_target
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--lr_revision', type=float, help='revision training learning rate', default=5e-7)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--model_dir', type=str, help='dir to save model files', default='model')
parser.add_argument('--prob_dir', type=str, help='dir to save output probability files', default='prob' )
parser.add_argument('--matrix_dir', type=str, help='dir to save estimated matrix', default='matrix')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--n_epoch_revision', type=int, default=200)
parser.add_argument('--n_epoch_estimate', type=int, default=20)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--percentile', type=int, default=97)
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()
args.noise_rate = '0.0'
#seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


#mnist, cifar10, cifar100
if args.dataset == 'mnist':
    args.n_epoch = 100
    args.n_epoch_estimate = 20
    args.num_classes = 10
    train_data = data_load.mnist_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                         noise_rate=args.noise_rate, random_seed=args.seed)
    val_data = data_load.mnist_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                       noise_rate=args.noise_rate, random_seed=args.seed)
    test_data = data_load.mnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    estimate_state = True
    model = Lenet.Lenet()
    
    
if args.dataset == 'cifar10':
    args.n_epoch = 200
    args.n_epoch_estimate = 20
    args.num_classes = 10
    #train_data = data_load.cifar10_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
    #                                     noise_rate=args.noise_rate, random_seed=args.seed)
    train_data = data_load.cifar10_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                         noise_rate=args.noise_rate, random_seed=args.seed)
    #val_data = data_load.cifar10_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
    #                                   noise_rate=args.noise_rate, random_seed=args.seed)
    test_data = data_load.cifar10_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    estimate_state = True
    model = Resnet.ResNet18(args.num_classes)
    
    
if args.dataset == 'cifar100':
    args.n_epoch = 200
    args.n_epoch_estimate = 15
    args.num_classes = 100
    train_data = data_load.cifar100_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                         noise_rate=args.noise_rate, random_seed=args.seed)
    val_data = data_load.cifar100_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                       noise_rate=args.noise_rate, random_seed=args.seed)
    test_data = data_load.cifar100_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    estimate_state = False
    model = Resnet.ResNet34(args.num_classes)
    
    
#optimizer and StepLR
optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
optimizer_revision = optim.Adam(model.parameters(), lr=args.lr_revision, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

      
    
#data_loader
train_loader = DataLoader(dataset=train_data, 
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=8,
                          drop_last=False)

estimate_loader = DataLoader(dataset=train_data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=8,
                             drop_last=False)

#val_loader = DataLoader(dataset=val_data,
#                        batch_size=args.batch_size,
#                        shuffle=False,
#                        num_workers=8,
#                        drop_last=False)

test_loader = DataLoader(dataset=test_data,
                         batch_size=args.batch_size,
                         num_workers=8,
                         drop_last=False)

#loss
loss_func_ce = nn.CrossEntropyLoss()
loss_func_reweight = reweight_loss()
loss_func_revision = reweighting_revision_loss()

#cuda
if torch.cuda.is_available:
    model = model.cuda()
    loss_func_ce = loss_func_ce.cuda()
    loss_func_reweight = loss_func_reweight.cuda()
    loss_func_revision = loss_func_revision.cuda()
    
#mkdir
model_save_dir = args.model_dir + '/' + args.dataset + '/' + 'noise_rate_%s'%(args.noise_rate) 

if not os.path.exists(model_save_dir):
    os.system('mkdir -p %s'%(model_save_dir))

prob_save_dir = args.prob_dir + '/' + args.dataset + '/' + 'noise_rate_%s'%(args.noise_rate)

if not os.path.exists(prob_save_dir):
    os.system('mkdir -p %s'%(prob_save_dir))

matrix_save_dir = args.matrix_dir + '/' + args.dataset + '/' + 'noise_rate_%s'%(args.noise_rate)

if not os.path.exists(matrix_save_dir):
    os.system('mkdir -p %s'%(matrix_save_dir))

#estimate transition matrix
index_num = int(len(train_data) / args.batch_size)
A = torch.zeros((args.n_epoch_estimate, len(train_data), args.num_classes))   
val_acc_list = []
total_index =  index_num + 1

#main function
def main():
    
    print('Estimate transition matirx......Waiting......')
    
    for epoch in range(args.n_epoch_estimate):
      
        print('epoch {}'.format(epoch + 1))
        model.train()
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
     
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer_es.zero_grad()
            out = model(batch_x, revision=False)
            loss = loss_func_ce(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer_es.step()
        
        torch.save(model.state_dict(), model_save_dir + '/'+ 'epoch_%d.pth'%(epoch+1))
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data))*args.batch_size, train_acc / (len(train_data))))
        
        with torch.no_grad():
            model.eval()
            #for batch_x, batch_y in val_loader:
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                out = model(batch_x, revision=False)
                loss = loss_func_ce(out, batch_y)
                val_loss += loss.item()
                pred = torch.max(out, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()
                
        #print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*args.batch_size, val_acc / (len(val_data)))) 
        #val_acc_list.append(val_acc / (len(val_data)))
        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(test_data))*args.batch_size, val_acc / (len(test_data)))) 
        val_acc_list.append(val_acc / (len(test_data)))

        
        
        #go through training data with model.eval, not shuffling data
        with torch.no_grad():
            model.eval()
            for index,(batch_x,batch_y) in enumerate(estimate_loader):
                 batch_x = batch_x.cuda()
                 out = model(batch_x, revision=False)
                 out = F.softmax(out,dim=1)
                 out = out.cpu()
                 if index <= index_num:
                    A[epoch][index*args.batch_size:(index+1)*args.batch_size, :] = out 
                 else:
                     A[epoch][index_num*args.batch_size, len(train_data), :] = out 
       
    val_acc_array = np.array(val_acc_list)
    model_index = np.argmax(val_acc_array)
    print('best test acc model_index', model_index)
    
    A_save_dir = prob_save_dir + '/' + 'prob.npy'
    np.save(A_save_dir, A)           

if __name__=='__main__':
    main()
