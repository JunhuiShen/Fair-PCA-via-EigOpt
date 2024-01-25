function loss = loss_trace(V,H)
loss = trace(V' * H * V);
end