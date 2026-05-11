
for batch in dataloader:
    context_tokens, target_future, actions = batch
    
    logits = model(context_tokens, actions)
    
    loss = F.cross_entropy(logits.view(-1, vocab_size), target_future.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()