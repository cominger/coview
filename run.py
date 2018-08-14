import numpy as np
import torch
import torch.nn as nn
import os
import pdb
from utils import progress_bar
from torch.autograd import Variable

best_acc = 0
train_acc = 0

# Training
def train_rgb(epoch, net, device ,optimizer, trainloader, criterion, args):

    global train_acc

    print('\nEpoch: %d' % epoch)
    scene_train_loss = 0
    action_train_loss = 0

    scene_correct = 0
    action_correct = 0
    scene_total = 0
    action_total = 0
    for batch_idx, (rgb, audio, scene_target, action_target) in enumerate(trainloader):
        
        scene_target -= 1
        action_target -= 30
        
        rgb, scene_target, action_target = rgb.to(device), scene_target.to(device), action_target.to(device)

        optimizer.zero_grad()
        scene_outputs, action_outputs = net(rgb)

        scene_loss = criterion(scene_outputs, scene_target)
        scene_loss.backward(retain_graph=True)

        action_loss = criterion(action_outputs, action_target)
        action_loss.backward()

        optimizer.step()

        scene_train_loss += scene_loss.item()
        action_train_loss += action_loss.item()

        _, scene_predicted = torch.max(scene_outputs.data, 1)
        _, action_predicted = torch.max(action_outputs.data, 1)

        scene_total += scene_target.size(0)
        action_total += action_target.size(0)
        
        scene_correct += scene_predicted.eq(scene_target.data).sum()
        action_correct += action_predicted.eq(action_target.data).sum()
        cur_scene = scene_predicted.eq(scene_target)
        cur_act = action_predicted.eq(action_target).sum(1).unsqueeze(1)
        cur_total = (cur_scene.byte() == cur_act.byte())
        total_correct += cur_total.sum() 

        progress_bar(batch_idx, len(trainloader), 'SceneLoss: %.3f , SceneAcc: %.3f%% (%d/%d) | ActionLoss: %.3f , ActionAcc: %.3f%% (%d/%d) | TotalLoss: %.3f | TotalAcc: %.3f%% (%d/%d)'
            % (scene_train_loss/(batch_idx+1), 100.*scene_correct/scene_total, scene_correct, scene_total,
               action_train_loss/(batch_idx+1), 100.*action_correct/action_total, action_correct, action_total,
               100.*float(total_correct)/float(total_total), total_correct, total_total))

    train_acc = 100. * float(total_correct)/float(total_total)

def test_rgb(epoch, net, device, testloader, criterion, args):
    global best_acc

    scene_test_loss = 0
    action_test_loss = 0

    scene_correct = 0
    action_correct = 0
    scene_total = 0
    action_total = 0

    with torch.no_grad():
        for batch_idx, (rgb, audio, scene_target, action_target) in enumerate(testloader):

            scene_target -= 1
            action_target -= 30
     
            rgb, scene_target, action_target = rgb.to(device), scene_target.to(device), action_target.to(device)
            scene_outputs, action_outputs = net(rgb)

            scene_loss = criterion(scene_outputs, scene_target)
            action_loss = criterion(action_outputs, action_target)

            scene_test_loss += scene_loss.item()
            action_test_loss += action_loss.item()

            _, scene_predicted = torch.max(scene_outputs.data, 1)
            _, action_predicted = torch.max(action_outputs.data, 1)

            scene_total += scene_target.size(0)
            action_total += action_target.size(0)

            scene_correct += scene_predicted.eq(scene_target.data).sum()
            action_correct += action_predicted.eq(action_target.data).sum()

            cur_scene = scene_predicted.eq(scene_target)
            cur_act = action_predicted.eq(action_target).sum(1).unsqueeze(1)
            cur_total = (cur_scene.byte() == cur_act.byte())
            total_correct += cur_total.sum() 

            progress_bar(batch_idx, len(trainloader), 'SceneLoss: %.3f , SceneAcc: %.3f%% (%d/%d) | ActionLoss: %.3f , ActionAcc: %.3f%% (%d/%d) | TotalLoss: %.3f | TotalAcc: %.3f%% (%d/%d)'
                % (scene_train_loss/(batch_idx+1), 100.*scene_correct/scene_total, scene_correct, scene_total,
                   action_train_loss/(batch_idx+1), 100.*action_correct/action_total, action_correct, action_total,
                   100.*float(total_correct)/float(total_total), total_correct, total_total))

    # Save checkpoint.
    acc = 100. * float(action_correct+scene_correct)/ float(action_total+scene_total)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict() if args.use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        filedir = './checkpoint/'
        state_file = filedir + args.filename + '_.pt'
        acc_file = filedir + args.filename + '_score.txt'
        torch.save(state, state_file)
        best_acc = acc
        File = open(acc_file, "w")
        File.write('Epoch: %d \n' % (epoch))
        File.write('Train Accuracy: %.3f %% \n' % (train_acc))
        File.write('Test Accuracy: %.3f %% \n' % (best_acc))
        File.close()
# Training
def train_audio(epoch, net, device, optimizer, trainloader, criterion, args):

    global train_acc


    print('\nEpoch: %d' % epoch)
    net.train()
    scene_train_loss = 0
    action_train_loss = 0

    scene_correct = 0
    action_correct = 0
    scene_total = 0
    action_total = 0
    for batch_idx, (rgb, audio, scene_target, action_target) in enumerate(trainloader):
        
        scene_target -= 1
        action_target -= 30
        
        audio, scene_target, action_target = audio.to(device), scene_target.to(device), action_target.to(device)

        optimizer.zero_grad()
        scene_outputs, action_outputs = net(audio)

        scene_loss = criterion(scene_outputs, scene_target)
        scene_loss.backward(retain_graph=True)

        action_loss = criterion(action_outputs, action_target)
        action_loss.backward()

        optimizer.step()

        scene_train_loss += scene_loss.item()
        action_train_loss += action_loss.item()

        _, scene_predicted = torch.max(scene_outputs.data, 1)
        _, action_predicted = torch.max(action_outputs.data, 1)

        scene_total += scene_target.size(0)
        action_total += action_target.size(0)
        
        scene_correct += scene_predicted.eq(scene_target.data).sum()
        action_correct += action_predicted.eq(action_target.data).sum()
        cur_scene = scene_predicted.eq(scene_target)
        cur_act = action_predicted.eq(action_target).sum(1).unsqueeze(1)
        cur_total = (cur_scene.byte() == cur_act.byte())
        total_correct += cur_total.sum() 

        progress_bar(batch_idx, len(trainloader), 'SceneLoss: %.3f , SceneAcc: %.3f%% (%d/%d) | ActionLoss: %.3f , ActionAcc: %.3f%% (%d/%d) | TotalLoss: %.3f | TotalAcc: %.3f%% (%d/%d)'
            % (scene_train_loss/(batch_idx+1), 100.*scene_correct/scene_total, scene_correct, scene_total,
               action_train_loss/(batch_idx+1), 100.*action_correct/action_total, action_correct, action_total,
               100.*float(total_correct)/float(total_total), total_correct, total_total))

    train_acc = 100. * float(total_correct)/float(total_total)

def test_audio(epoch, net, device,  testloader, criterion, args):
    global best_acc

    net.eval()
    scene_test_loss = 0
    action_test_loss = 0

    scene_correct = 0
    action_correct = 0
    scene_total = 0
    action_total = 0
    with torch.no_grad():
        for batch_idx, (rgb, audio, scene_target, action_target) in enumerate(testloader):

            scene_target -= 1
            action_target -= 30
     
            audio, scene_target, action_target = audio.to(device), scene_target.to(device), action_target.to(device)        
            scene_outputs, action_outputs = net(audio)

            scene_loss = criterion(scene_outputs, scene_target)
            action_loss = criterion(action_outputs, action_target)

            scene_test_loss += scene_loss.item()
            action_test_loss += action_loss.item()

            _, scene_predicted = torch.max(scene_outputs.data, 1)
            _, action_predicted = torch.max(action_outputs.data, 1)

            scene_total += scene_target.size(0)
            action_total += action_target.size(0)

            scene_correct += scene_predicted.eq(scene_target.data).sum()
            action_correct += action_predicted.eq(action_target.data).sum()

            cur_scene = scene_predicted.eq(scene_target)
            cur_act = action_predicted.eq(action_target).sum(1).unsqueeze(1)
            cur_total = (cur_scene.byte() == cur_act.byte())
            total_correct += cur_total.sum() 

            progress_bar(batch_idx, len(trainloader), 'SceneLoss: %.3f , SceneAcc: %.3f%% (%d/%d) | ActionLoss: %.3f , ActionAcc: %.3f%% (%d/%d) | TotalLoss: %.3f | TotalAcc: %.3f%% (%d/%d)'
                % (scene_train_loss/(batch_idx+1), 100.*scene_correct/scene_total, scene_correct, scene_total,
                   action_train_loss/(batch_idx+1), 100.*action_correct/action_total, action_correct, action_total,
                   100.*float(total_correct)/float(total_total), total_correct, total_total))

        # Save checkpoint.
        acc = 100. * float(action_correct+scene_correct)/ float(action_total+scene_total)
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict() if args.use_cuda else net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
            filedir = './checkpoint/'
            state_file = filedir + args.filename + '_.pt'
            acc_file = filedir + args.filename + '_score.txt'
            torch.save(state, state_file)
            best_acc = acc
            File = open(acc_file, "w")
            File.write('Epoch: %d \n' % (epoch))
            File.write('Train Accuracy: %.3f %% \n' % (train_acc))
            File.write('Test Accuracy: %.3f %% \n' % (best_acc))
            File.close()

def two_stream(net_audio, net_rgb,  device, dataloader, args):
    global best_acc

    scene_correct = 0
    action_correct = 0
    total_correct =0
    scene_total = 0
    action_total = 0
    total_total = 0

    top_k = 1
    fin_id = []
    fin_scene_pred=[]
    fin_action_pred=[]

    with torch.no_grad():
        for batch_idx, (rgb, audio, scene_target, action_target, ids) in enumerate(dataloader):

            scene_target -= 1
            action_target -= 30

            audio, rgb, scene_target, action_target = audio.to(device), rgb.to(device), scene_target.to(device), action_target.to(device)
            
            m=nn.Softmax(dim=1)
            audio_scene_outputs, audio_action_outputs = net_audio(audio)
            rgb_scene_outputs, rgb_action_outputs = net_rgb(rgb)

            scene_outputs = (audio_scene_outputs.cpu()) +  (rgb_scene_outputs.cpu())      
            action_outputs = (audio_action_outputs.cpu()) + (rgb_action_outputs.cpu())

            _, scene_predicted = torch.topk(scene_outputs.data, top_k, dim=1)
            _, action_predicted = torch.topk(action_outputs.data, top_k, dim=1)

            scene_target  = torch.reshape(scene_target.unsqueeze(1).repeat(1, top_k), scene_predicted.shape)
            action_target = torch.reshape(action_target.unsqueeze(1).repeat(1, top_k), action_predicted.shape)

            ids = np.asarray(ids)
            ids = np.expand_dims(ids, axis=1)
            ori_scene_predicted = (scene_predicted + 1).cpu().numpy()
            ori_action_predicted = (action_predicted + 30).cpu().numpy()

            if not batch_idx:
                fin_id = ids
                fin_scene_pred = ori_scene_predicted
                fin_action_pred = ori_action_predicted

            else:
                fin_id = np.vstack((fin_id, ids))
                fin_scene_pred = np.vstack((fin_scene_pred, ori_scene_predicted))
                fin_action_pred = np.vstack((fin_action_pred, ori_action_predicted))

            scene_total += scene_target.size(0)
            action_total += action_target.size(0)
            total_total += action_target.size(0)

            scene_correct += scene_predicted.eq(scene_target.data).sum()
            action_correct += action_predicted.eq(action_target.data).sum()
            cur_scene = scene_predicted.eq(scene_target)
            cur_act = action_predicted.eq(action_target).sum(1).unsqueeze(1)
            cur_total = (cur_scene.byte() == cur_act.byte())
            total_correct += cur_total.sum() 

            progress_bar(batch_idx, len(dataloader), 'SceneAcc: %.3f%% (%d/%d) | ActionAcc: %.3f%% (%d/%d) | TotalAcc: %.3f%% (%d/%d)'
                % ( 100.*float(scene_correct)/float(scene_total), scene_correct, scene_total,
                    100.*float(action_correct)/float(action_total), action_correct, action_total,
                    100.*float(total_correct)/float(total_total), total_correct, total_total))

#    fin_total=np.hstack((fin_id, fin_scene_pred, fin_action_pred))
#    fin_total = fin_total.astype(object)
#    fmt = '%s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d'
#    np.savetxt("test.csv", fin_total, fmt=fmt)

def two_stream_comb(net_audio, net_rgb, dataloader, args):
    global best_acc

    net_audio.eval()
    net_rgb.eval()

    scene_correct = 0
    action_correct = 0
    scene_total = 0
    action_total = 0

    top_k = 5
    weight = [x*0.01 for x in range(101)]
    for per in weight:
        a_weight = per
        r_weight = 1.0-per
        print("\n (%5.3f, %5.3f)" %( a_weight,r_weight))
     
        scene_correct = 0
        action_correct = 0
        scene_total = 0
        action_total = 0

        for batch_idx, (rgb, audio, scene_target, action_target) in enumerate(dataloader):

            scene_target -= 1
            action_target -= 30
 

            if args.use_cuda:
                audio, rgb = audio.cuda(), rgb.cuda()
            with torch.no_grad():
                audio, rgb, scene_target, action_target = Variable(audio), Variable(rgb), Variable(scene_target), Variable(action_target)
        
                m=nn.Softmax(dim=1)
                audio_scene_outputs, audio_action_outputs = net_audio(audio)
                rgb_scene_outputs, rgb_action_outputs = net_rgb(rgb)

                scene_outputs = torch.mul((audio_scene_outputs.cpu()), a_weight) +  torch.mul((rgb_scene_outputs.cpu()), r_weight)
                #scene_outputs = (audio_scene_outputs.cpu()) +  (rgb_scene_outputs.cpu())
                #scene_outputs = m(rgb_scene_outputs.cpu())
       
                action_outputs = torch.mul((audio_action_outputs.cpu()), a_weight) + torch.mul((rgb_action_outputs.cpu()), r_weight)
                #action_outputs = (audio_action_outputs.cpu()) + (rgb_action_outputs.cpu())
                #action_outputs = m(rgb_action_outputs.cpu())

                #_, scene_predicted = torch.max(scene_outputs.data, 1)
                _, scene_predicted = torch.topk(scene_outputs.data, top_k, dim=1)
                #_, action_predicted = torch.max(action_outputs.data, 1)
                _, action_predicted = torch.topk(action_outputs.data, top_k, dim=1)
 
                scene_target  = torch.reshape(scene_target.unsqueeze(1).repeat(1,top_k), scene_predicted.shape)
                action_target = torch.reshape(action_target.unsqueeze(1).repeat(1,top_k), action_predicted.shape)

                scene_total += scene_target.size(0)
                action_total += action_target.size(0)

                scene_correct += scene_predicted.eq(scene_target.data).sum()
                action_correct += action_predicted.eq(action_target.data).sum()

                progress_bar(batch_idx, len(dataloader), 'SceneAcc: %.3f%% (%d/%d) | ActionAcc: %.3f%% (%d/%d) | TotalAcc: %.3f%% (%d/%d)'
                    % ( 100.*float(scene_correct)/float(scene_total), scene_correct, scene_total,
                        100.*float(action_correct)/float(action_total), action_correct, action_total,
                        100. * float(action_correct+scene_correct)/ float(action_total+scene_total), action_correct+scene_correct, action_total+scene_total))

