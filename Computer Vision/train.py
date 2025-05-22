def train(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        for img, mask in dataloader:
            img, mask = img.to(device), mask.to(device)

            pred = model(img)
            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Epoch {epoch}] Loss: {epoch_loss / len(dataloader):.4f}")
