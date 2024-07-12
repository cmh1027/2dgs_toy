# color using simultaneous equation 
# numer, denom = 0, 0
# for _ in trange(25):
    # X = torch.zeros(N, N).cuda()
    # Y = torch.zeros(N, 3).cuda()
    # for i in trange(len(data), leave=False):
    #     normal, bias, y = data[i] # y : (k, 3)
    #     y_pred, weight = model.render(x, normal, bias) # weight : (N, k, 1)
    #     Y += (weight * y[None, ...].repeat(N, 1, 1)).sum(dim=1)
    #     X += (weight.permute(1, 0, 2) * weight.permute(1, 2, 0)).sum(dim=0) # (k, N, 1) * (k, 1, N) = (k, N, N) => (N, N)
    # result = torch.matmul(torch.inverse(X), Y)

                    # if 'opacity' not in fixed:
                    #     c = model.get_rgb
                    #     c_pad = torch.zeros_like(model.get_rgb)
                    #     c_pad[:-1] = c[1:]
                    #     T_pad = torch.zeros_like(T)
                    #     T_pad[:-1] = T[1:]
                    #     (T * c - T_pad * c_pad) / (T * g * c)