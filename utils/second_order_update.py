import torch
import torch.nn
import torch.nn.functional as F


def update(model, updated_model, criterion, optimizer_w,
           inputs_val, targets_val, inputs_train, targets_train):
  # extract some hyper-parameters
  lr = optimizer_w.param_groups[0]['lr']
  wd = optimizer_w.param_groups[0]['weight_decay']
  momentum = optimizer_w.param_groups[0]['momentum']

  # -------------------------------------------------------------------------------------
  # -----------------------step1: get an updated model w.r.t train loss------------------
  # -------------------------------------------------------------------------------------

  # forward & calc loss
  loss = criterion(model(inputs_train), targets_train)  # L_train(w)

  # compute gradient
  weights = [v for k, v in model.named_parameters() if 'alpha' not in k]
  new_weights = [v for k, v in updated_model.named_parameters() if 'alpha' not in k]
  gradients = torch.autograd.grad(loss, weights)

  # do virtual step (update gradient)
  # below operations do not need gradient tracking
  with torch.no_grad():
    # optimizer.state is a dict, which uses model's parameters as keys
    # however, the dict key is not the value, but the pointer.
    # so original network weight have to be iterated also.
    for w, new_w, grad in zip(weights, new_weights, gradients):
      mom = optimizer_w.state[w].get('momentum_buffer', 0.) * momentum
      new_w.copy_(w - lr * (mom + grad + wd * w))

    alphas = [v for k, v in model.named_parameters() if 'alpha' in k]
    new_alphas = [v for k, v in updated_model.named_parameters() if 'alpha' in k]
    # simply copy the value of alphas
    for a, new_a in zip(alphas, new_alphas):
      new_a.copy_(a)

  # -------------------------------------------------------------------------------------
  # ------------------step2: get dL_val(w', a)/dw' and dL_val(w', a)/da------------------
  # -------------------------------------------------------------------------------------

  # calc val loss on updated model
  val_loss = criterion(updated_model(inputs_val), targets_val)  # L_val(w', a)

  # compute gradient
  grad_new = torch.autograd.grad(val_loss, new_alphas + new_weights)
  grad_new_alphas = grad_new[:len(new_alphas)]
  grad_new_weights = grad_new[len(new_alphas):]

  # -------------------------------------------------------------------------------------
  # ---------------------------step3: compute approximated hessian-----------------------
  # -------------------------------------------------------------------------------------

  # dw = dw' { L_val(w', a) }
  # w+ = w + eps * dw
  # w- = w - eps * dw
  # hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)

  # eps = 0.01 / ||dw||
  norm = torch.cat([g.view(-1) for g in grad_new_weights]).norm()
  eps = 0.01 / norm

  # w+ = w + eps*dw'
  with torch.no_grad():
    for w, grad in zip(weights, grad_new_weights):
      w += eps * grad

  loss = criterion(model(inputs_train), targets_train)  # L_train(w+)
  grad_alphas_pos = torch.autograd.grad(loss, alphas)  # dalpha { L_train(w+) }

  # w- = w - eps*dw'
  with torch.no_grad():
    for w, grad in zip(weights, grad_new_weights):
      w -= 2. * eps * grad

  loss = criterion(model(inputs_train), targets_train)  # L_train(w-)
  grad_alphas_neg = torch.autograd.grad(loss, alphas)  # dalpha { L_train(w-) }

  # recover w
  with torch.no_grad():
    for w, grad in zip(weights, grad_new_weights):
      w += eps * grad

  hessian = [(g_pos - g_neg) / 2. * eps for g_pos, g_neg
             in zip(grad_alphas_pos, grad_alphas_neg)]

  # -------------------------------------------------------------------------------------
  # -----------------------------------step4: update alphas------------------------------
  # -------------------------------------------------------------------------------------

  # update final gradient = dalpha - xi*hessian
  with torch.no_grad():
    for a, grad_a, h in zip(alphas, grad_new_alphas, hessian):
      a.grad = grad_a - lr * h

  return val_loss


def alpha_entropy_grad(alphas):
  with torch.no_grad():
    probs = F.softmax(alphas, dim=1)
    dw = -torch.bmm(probs[:, :, None], probs[:, None, :])
    dw[:, torch.arange(alphas.shape[1]), torch.arange(alphas.shape[1])] += probs
    grad = (-dw * (torch.log(probs[:, :, None]) + 1)).sum(1)
  return grad


def alpha_entropy_loss(alphas, axis=-1):
  probs = F.softmax(alphas, dim=axis)
  entropy = ((-probs * probs.log()).sum(axis)).mean()
  return entropy

# if __name__ == '__main__':
#   alpha_entropy_grad(torch.randn(14,8))

# if __name__ == '__main__':
#   from copy import deepcopy
#   from nets.cifar_search_model import *
#
#   model = Network(6, 8, 4)
#   temp_model = Network(6, 8, 4)
#
#   alphas=[v for k,v in model.named_parameters() if 'alpha' in k]
#   weights = [v for k, v in model.named_parameters() if 'alpha' not in k]
#   optimizer = torch.optim.SGD(weights, lr=0.01, momentum=0.9, weight_decay=5e-4)
#   alpha_optim = torch.optim.Adam(alphas, 1e-2, betas=(0.5, 0.999), weight_decay=1e-4)
#   criterion = nn.CrossEntropyLoss()
#
#   loss = criterion(model(torch.randn(10, 3, 32, 32)), torch.randint(0, 10, [10]).long())
#   loss.backward()
#   optimizer.step()
#
#   alpha_optim.zero_grad()
#   before = deepcopy(model)
#   update(model, temp_model, criterion, optimizer,
#          torch.randn(10, 3, 32, 32), torch.randint(0, 10, [10]).long(),
#          torch.randn(10, 3, 32, 32), torch.randint(0, 10, [10]).long())
#   alpha_optim.step()
#   after = model
