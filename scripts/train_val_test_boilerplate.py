import numpy as np
import pandas as pd
import gzip
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import time
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from utils import get_CKA, get_r2_score, get_rmse_score


def init_valid_loss(model, val_loader, g=[0,1,2,3,4], h=[5,6,7,8,9], device='cpu'):
    print('Computing initial validation loss')
    model.eval()
    valid_loss1, valid_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
    min_valid_loss, points = 0.0, 0
    for x, y, _ in val_loader:
        with torch.no_grad():
            bs   = x.shape[0]                #batch size
            x    = x.to(device=device)       #maps
            y    = y.to(device=device)[:,g]  #parameters
            p    = model(x)                  #NN output
            y_NN = p[:,g]                    #posterior mean
            e_NN = p[:,h]                    #posterior std
            loss1 = torch.mean((y_NN - y)**2,                axis=0)
            loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
            loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
            valid_loss1 += loss1*bs
            valid_loss2 += loss2*bs
            points += bs
    min_valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
    min_valid_loss = torch.mean(min_valid_loss).item()
    print('Initial valid loss = %.3e'%min_valid_loss)
    return min_valid_loss


def train(
        model, train_loader, val_loader, epochs, optimizer, scheduler, min_valid_loss,
        fmodel='weights.pt', floss='loss.txt', g=[0,1,2,3,4], h=[5,6,7,8,9], device='cpu'
):
    slopes_omega_m = []
    slopes_sigma_8 = []

    # do a loop over all epochs
    start = time.time()
    for epoch in range(epochs):
        cka_mats_val = []

        # do training
        train_loss1, train_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
        train_loss, points = 0.0, 0
        model.train()
        for x, y, _ in train_loader:
            bs   = x.shape[0]         #batch size
            x    = x.to(device)       #maps
            y    = y.to(device)[:,g]  #parameters
            p    = model(x)           #NN output
            y_NN = p[:,g]             #posterior mean
            e_NN = p[:,h]             #posterior std
            loss1 = torch.mean((y_NN - y)**2,                axis=0)
            loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
            loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
            train_loss1 += loss1*bs
            train_loss2 += loss2*bs
            points      += bs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if points>18000:  break
        train_loss = torch.log(train_loss1/points) + torch.log(train_loss2/points)
        train_loss = torch.mean(train_loss).item()

        # do validation: cosmo alone & all params
        valid_loss1, valid_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
        valid_loss, points = 0.0, 0
        val_true_params = []
        val_pred_params = []
        model.eval()
        for x, y, _ in val_loader:
            with torch.no_grad():
                bs    = x.shape[0]         #batch size
                x     = x.to(device)       #maps
                y     = y.to(device)[:,g]  #parameters
                p     = model(x)           #NN output
                y_NN  = p[:,g]             #posterior mean
                e_NN  = p[:,h]             #posterior std
                loss1 = torch.mean((y_NN - y)**2,                axis=0)
                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
                valid_loss1 += loss1*bs
                valid_loss2 += loss2*bs
                points     += bs

                # CKA on validation set
                mid_getter = MidGetter(model, return_layers={'LeakyReLU': 'LeakyReLU'}, keep_output=True)
                mid_outputs1 = mid_getter(x)
                mid_outputs2 = mid_getter(x)

                intermediate_outputs_A = mid_outputs1[0]['LeakyReLU']
                intermediate_outputs_B = mid_outputs2[0]['LeakyReLU']

                intermediate_outputs_A = [o.cpu() for o in intermediate_outputs_A]
                intermediate_outputs_B = [ob.cpu() for ob in intermediate_outputs_B]

                sim = get_CKA(n_layers=6, n_layers2=6, activations1=intermediate_outputs_A, activations2=intermediate_outputs_B)  # change 14 to some other number if no. of layers are different.
                cka_mats_val.append(sim)

                val_true_params.append(y)
                val_pred_params.append(y_NN)

        val_true_params = torch.vstack(val_true_params)
        val_pred_params = torch.vstack(val_pred_params)

        omega_m_true = val_true_params[:, 0]
        sigma_8_true = val_true_params[:, 4]
        omega_m_pred = val_pred_params[:, 0]
        sigma_8_pred = val_pred_params[:, 4]

        slope_omega_m, _ = np.polyfit(omega_m_true.cpu().numpy(), omega_m_pred.cpu().numpy(), 1)
        slope_sigma_8, _ = np.polyfit(sigma_8_true.cpu().numpy(), sigma_8_pred.cpu().numpy(), 1)
        slopes_omega_m.append(slope_omega_m)
        slopes_sigma_8.append(slope_sigma_8)

        valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
        valid_loss = torch.mean(valid_loss).item()

        scheduler.step(valid_loss)

        # verbose
        print('%03d %.3e %.3e '%(epoch, train_loss, valid_loss), end='')

        # save model if it is better
        if valid_loss<min_valid_loss:
            torch.save(model.state_dict(), fmodel)
            min_valid_loss = valid_loss
            print('(C) ', end='')
        print('')

        # save losses to file
        f = open(floss, 'a')
        f.write('%d %.5e %.5e\n'%(epoch, train_loss, valid_loss))
        f.close()

        # Save the averaged CKA matrix across the entire val set.
        final_cka = np.stack(cka_mats_val).mean(axis=0)
        np.save(f'cka_epoch{epoch}_val.npy', final_cka)

    stop = time.time()
    print('Time take (h):', "{:.4f}".format((stop-start)/3600.0))

    return model


def test(model, test_loader, g=[0,1,2,3,4], h=[5,6,7,8,9], device='cpu'):
    # get the number of maps in the test set
    num_maps = 0
    for x, y, _ in test_loader:
        num_maps += x.shape[0]
    print('\nNumber of maps in the test set: %d'%num_maps)

    # define the arrays containing the value of the parameters
    params_true = np.zeros((num_maps,5), dtype=np.float32)
    params_NN   = np.zeros((num_maps,5), dtype=np.float32)
    errors_NN   = np.zeros((num_maps,5), dtype=np.float32)
    filenames = np.empty((num_maps), dtype='object')

    # get test loss
    test_loss1, test_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
    test_loss, points = 0.0, 0
    model.eval()
    for x, y, filename in test_loader:
        with torch.no_grad():
            bs    = x.shape[0]    #batch size
            x     = x.to(device)  #send data to device
            y     = y.to(device)  #send data to device
            p     = model(x)      #prediction for mean and variance
            y_NN  = p[:,:5]       #prediction for mean
            e_NN  = p[:,5:]       #prediction for error
            loss1 = torch.mean((y_NN[:,g] - y[:,g])**2,                     axis=0)
            loss2 = torch.mean(((y_NN[:,g] - y[:,g])**2 - e_NN[:,g]**2)**2, axis=0)
            test_loss1 += loss1*bs
            test_loss2 += loss2*bs

            # save results to their corresponding arrays
            params_true[points:points+x.shape[0]] = y.cpu().numpy()
            params_NN[points:points+x.shape[0]]   = y_NN.cpu().numpy()
            errors_NN[points:points+x.shape[0]]   = e_NN.cpu().numpy()
            filenames[points:points+x.shape[0]]   = filename
            points    += x.shape[0]

    test_loss = torch.log(test_loss1/points) + torch.log(test_loss2/points)
    test_loss = torch.mean(test_loss).item()
    print('Test loss = %.3e\n'%test_loss)

    Norm_error = np.sqrt(np.mean((params_true - params_NN)**2, axis=0))
    print('Normalized Error Omega_m = %.3f'%Norm_error[0])
    print('Normalized Error Omega_b = %.3f'%Norm_error[1])
    print('Normalized Error h   = %.3f'%Norm_error[2])
    print('Normalized Error n_s  = %.3f'%Norm_error[3])
    print('Normalized Error sigma_8   = %.3f'%Norm_error[4])

    # de-normalize
    # IMPORTANT: These values must match the ones used during normalization in the preprocessing step before training.
    minimum = np.array([0.1003, 0.03003, 0.5003, 0.8001, 0.6001])
    maximum = np.array([0.4997, 0.06993, 0.8999, 1.1999, 0.9985])
    params_true = params_true*(maximum - minimum) + minimum
    params_NN   = params_NN*(maximum - minimum) + minimum
    errors_NN   = errors_NN*(maximum - minimum)

    error = np.sqrt(np.mean((params_true - params_NN)**2, axis=0))
    print('Error Omega_m = %.3f'%error[0])
    print('Error Omega_b = %.3f'%error[1])
    print('Error h   = %.3f'%error[2])
    print('Error n_s  = %.3f'%error[3])
    print('Error sigma_8   = %.3f'%error[4])

    mean_error = np.absolute(np.mean(errors_NN, axis=0))
    print('Bayesian error Omega_m = %.3f'%mean_error[0])
    print('Bayesian error Omega_b = %.3f'%mean_error[1])
    print('Bayesian error h   = %.3f'%mean_error[2])
    print('Bayesian error n_s  = %.3f'%mean_error[3])
    print('Bayesian error sigma_8   = %.3f'%mean_error[4])

    rel_error = np.sqrt(np.mean((params_true - params_NN)**2/params_true**2, axis=0))
    print('Relative error Omega_m = %.3f'%rel_error[0])
    print('Relative error Omega_b = %.3f'%rel_error[1])
    print('Relative error h   = %.3f'%rel_error[2])
    print('Relative error n_s  = %.3f'%rel_error[3])
    print('Relative error sigma_8   = %.3f'%rel_error[4])


    # Save R2 and RMSE scores of each parameter
    r2_score = get_r2_score(params_true, params_NN)
    rmse_score = get_rmse_score(params_true, params_NN)

    np.save('r2_score_test.npy', r2_score)
    np.save('rmse_score_test.npy', rmse_score)

    # save results to file
    #dataset = np.zeros((num_maps,18), dtype=np.float32)
    #dataset[:,:6]   = params_true
    #dataset[:,6:12] = params_NN
    #dataset[:,12:]  = errors_NN
    #np.savetxt(fresults,  dataset)
    #np.savetxt(fresults1, Norm_error)