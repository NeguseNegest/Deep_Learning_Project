Models are saved as:

torch.save({"ddpmpp" : vp_net.state_dict()
                  , "ema_net" : ema_net.state_dict()
                  , "optimizer" : optimizer.state_dict()
                  , "loss_hist" : loss_hist
                  , "epochs" : epochs
                  }, path + f"/bird_ddpmpp_ema_checkpoint_epoch_{epoch_str}.pth")

To run, ...
