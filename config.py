### Original ###
# hyperparams = {
#     "hidden_dimension": 256,
#     "mlp_num": 8,
#     "res_at": 5,
#     "position_L": 10,
#     "angle_L": 4,
#     "position_scale": 4.0
#     "t_near": 2.0,
#     "t_far": 6.0,
#     "w_pixel_num": 800,
#     "batch_size": 4096,
#     "sample_coarse": 64,
#     "sample_fine": 128,
#     "epochs": 13, # 200K * 4096 / (100 * 800 * 800) 
#     "lr": 5e-4,
#     "decay_to": 5e-5,
# }

hyperparams = {
    "hidden_dimension": 256, #
    "mlp_num": 3, #
    "res_at": 0, #
    "position_L": 10, #
    "angle_L": 4, #
    "position_scale": 0.25,
    "t_near": 2.0,
    "t_far": 6.0,
    "w_pixel_num": 100, #
    "batch_size": 1024,
    "sample_coarse": 64,
    "sample_fine": 128, #
    "epochs": 20, # don't forget to adjust
    "lr": 5e-4, # don't forget to adjust
    "decay_to": 1e-4, # don't forget to adjust
}