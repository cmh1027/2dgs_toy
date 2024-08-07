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


    # elif METHOD == "BFGS":
    #     stack = list(range(len(images)))
    #     random.shuffle(images)
    #     optimizers = {}
    #     for param in ['rgb', 'xy', 'scale', 'rotation', 'opacity']:
    #         if param not in fixed:
    #             optimizer = torch.optim.LBFGS([getattr(model, "_" + param)], lr=1, max_iter=50000, max_eval=50000, history_size=50)
    #             optimizers[param] = optimizer

    #     for i in trange(100):
    #         for param in ['rgb', 'xy', 'scale', 'rotation', 'opacity']:
    #             if param in fixed: continue
    #             optimizers[param].zero_grad()
    #             def closure():
    #                 ys, y_preds, = [], []
    #                 for i in range(len(images)):
    #                     normal, bias, y = images[i]
    #                     y_pred, _ = model.render(data, normal, bias)
    #                     ys.append(y)
    #                     y_preds.append(y_pred)
    #                 y = torch.cat(ys, dim=0)
    #                 y_pred = torch.cat(y_preds, dim=0)
    #                 loss = ((y - y_pred)**2).mean()
    #                 loss.backward()
    #                 return loss
    #             optimizers[param].step(closure)


    # elif METHOD == "BFGS":
    #     fixed = []
    #     gt = True


            # def g(x):
            #     x_A_repeat = x[None, None, ...].repeat(3, N, 1, 1) # (3, N, N, 1)
            #     x_b_repeat = x[None, ...].repeat(3, 1, 1) # (3, N, 1)
            #     return mm(mm(x_A_repeat.permute(0, 1, 3, 2), A), x_A_repeat)[..., 0] + mm(b, x_b_repeat) + z[..., None] # (3, Ni, 1)
            # def h(x):
            #     x_A_repeat = x[None, None, ...].repeat(3, N, 1, 1) # (3, N, N, 1)
            #     return 2 * mm(A, x_A_repeat) + b[..., None] # (3, N, N, 1)
            # def F(x):
            #     g_x, h_x = g(x), h(x)
            #     return (2 * g_x.unsqueeze(2) * h_x).sum(dim=1) # (3, Nj, 1)
            # def J(x):
            #     g_x, h_x = g(x), h(x)
            #     return (4 * g_x[..., None] * A + 2 * mm(h_x, h_x.permute(0, 1, 3, 2))).sum(dim=1)  # (3, Nj, Nj)

            # damp = 0.02
            # tolerance = 1e-4
            # x = (torch.rand(N, 1) * 0.9 + 0.1).cuda()
            # # x = torch.ones(N, 1).cuda()
            # for i in trange(1000, desc="LM Method", leave=False):
            #     F_x = F(x)
            #     J_x = J(x)
            #     JTJ = mm(J_x.permute(0, 2, 1), J_x) # (3, N, N)
            #     M = JTJ + damp * torch.diag_embed(torch.diagonal(JTJ, dim1=1, dim2=2)) # (3, N, N)
            #     gradient = mm(J_x.permute(0, 2, 1), F_x)
            #     step = -mm(torch.inverse(M), gradient).mean(dim=0) # (N, 1)
            #     x_new = x + step
            #     F_x_new = F(x_new)
            #     norm = F_x.mean(dim=0).norm()
            #     norm_new = F_x_new.mean(dim=0).norm()
            #     if norm > norm_new:
            #         x = x_new
            #         damp = damp / 1.1
            #     else:
            #         damp = damp * 1.1
            #     if norm < tolerance:
            #         ic(norm.item())
            #         break
            #     if torch.norm(step) < tolerance:
            #         ic(torch.norm(step).item())
            #         break
            #     if torch.norm(gradient) < tolerance:
            #         ic(torch.norm(gradient).item())
            #         break
            # result_opacity = x


            # if i % 3 == 0:
            #     with torch.no_grad():
            #         ys, y_preds = [], []
            #         for i in trange(len(images), leave=False):
            #             normal, bias, y = images[i] # y : (k, 3)
            #             ys.append(y)
            #             y_pred, params = model.render(data, normal, bias, precomp_opacity=x)
            #             y_preds.append(y_pred)
            #         y = torch.cat(ys, dim=0)
            #         y_pred = torch.cat(y_preds, dim=0)
            #         ic(((y - y_pred) ** 2).mean().item())


                # def g(x, A, b, z):
                #     x_A_repeat = x[None, ..., None].repeat(3, N, 1, 1) # (3, N, N, 1)
                #     x_b_repeat = x[..., None].repeat(3, 1, 1) # (3, N, 1)
                #     return mm(mm(x_A_repeat.permute(0, 1, 3, 2), A), x_A_repeat)[..., 0] + mm(b, x_b_repeat) + z[..., None] # (3, Ni, 1)
                # def h(x, A, b):
                #     x_A_repeat = x[None, ..., None].repeat(3, N, 1, 1) # (3, N, N, 1)
                #     return 2 * mm(A, x_A_repeat) + b[..., None] # (3, N, N, 1)
                # def F(x, A, b, z):
                #     g_x = g(x, A, b, z)
                #     return (g_x ** 2).sum(dim=1) # (3, 1)
                # def J(x, A, b, z):
                #     g_x, h_x = g(x, A, b, z), h(x, A, b)
                #     return (2 * g_x.unsqueeze(2) * h_x).sum(dim=1) # (3, Nj, 1)

                # for i in trange(1000, desc="LM Method", leave=False):
                #     T_ = torch.ones_like(g_) 
                #     T_[0, 1:] = torch.cumprod(1 - x[..., None] * g_, dim=1)[0, :-1] # (1, N, k)
                #     P = T_ * g_ * c_.unsqueeze(2) # (3, Nj, k)
                #     dT_do = -T_.unsqueeze(1) * (g_ / (1 - g_ * x[..., None])).unsqueeze(2) # (1, Ni, Nj, k)
                #     dT_do = dT_do.permute(0, 3, 1, 2).triu(diagonal=1).permute(0, 2, 3, 1) # (1, Ni, Nj, k)

                #     Q = dT_do * g_.unsqueeze(2) * c_[..., None, None] # (3, Ni, Nj, k)
                #     P_repeat = P.unsqueeze(1).repeat(1, N, 1, 1) # (3, Ni, Nj, k)
                #     A = 0.5 * (mm(P_repeat, Q.permute(0, 1, 3, 2)) + mm(Q, P_repeat.permute(0, 1, 3, 2))) # (3, Ni, Nj, Nj)
                #     b = mm(P, P.permute(0, 2, 1)) - (y_[:, None, None, :] * Q).sum(dim=-1) # (3, Ni, Nj)
                #     z = -(y_.unsqueeze(1) * T_ * g_ * c_.unsqueeze(2)).sum(-1) # (3, Ni)
                    
                
                #     F_x = F(x, A, b, z) # (3, 1)
                #     J_x = J(x, A, b, z).squeeze(-1) # (3, Nj)
                #     JTJ = (J_x * J_x).sum(dim=-1, keepdim=True) # (3, 1)
                #     M = (damp+1) * JTJ # (3, 1)
                #     gradient = J_x * F_x # (3, Nj)
                #     step = (-gradient / M).mean(dim=0)[..., None] # (N, 1)
                #     x_new = x + step.permute(1, 0)
                #     F_x_new = F(x_new, A, b, z)
                #     norm = F_x.mean(dim=0)
                #     norm_new = F_x_new.mean(dim=0)

                #     if norm > norm_new:
                #         x = x_new
                #         damp = damp / 1.5
                #     else:
                #         damp = damp * 1.5
                #     if norm < tolerance:
                #         ic(norm.item())
                #         break
                #     if torch.norm(step) < tolerance:
                #         ic(torch.norm(step).item())
                #         break
                #     if torch.norm(gradient) < tolerance:
                #         ic(torch.norm(gradient).item())
                #         break

                # result_opacity = x.permute(1, 0)
                # model.set_opacity(result_opacity)



            # y_ = y.permute(1, 0) # (3, k)
            # c_ = model.get_rgb.permute(1, 0) # (3, N)
            # g_ = params['g'].permute(2, 0, 1) # (1, N, k)
            # T_ = params['T'].permute(2, 0, 1)
            # P = T_ * g_ * c_.unsqueeze(2) # (3, Nj, k)
            # o = model.get_opacity.permute(1, 0) # (1, N)
            # dT_do = -T_.unsqueeze(1) * (g_ / (1 - g_ * o[..., None])).unsqueeze(2) # (1, Ni, Nj, k)
            # dT_do = dT_do.permute(0, 3, 1, 2).triu(diagonal=1).permute(0, 2, 3, 1) # (1, Ni, Nj, k)
            # Q = dT_do * g_.unsqueeze(2) * c_[..., None, None] # (3, Ni, Nj, k)
            # P_repeat = P.unsqueeze(1).repeat(1, N, 1, 1) # (3, Ni, Nj, k)
            # A = 0.5 * (mm(P_repeat, Q.permute(0, 1, 3, 2)) + mm(Q, P_repeat.permute(0, 1, 3, 2))) # (3, Ni, Nj, Nj)
            # b = mm(P, P.permute(0, 2, 1)) - (y_[:, None, None, :] * Q).sum(dim=-1) # (3, Ni, Nj)
            # z = -(y_.unsqueeze(1) * T_ * g_ * c_.unsqueeze(2)).sum(-1) # (3, Ni)
            # ic(A[0, 0].mean(), b[0, 0].mean())