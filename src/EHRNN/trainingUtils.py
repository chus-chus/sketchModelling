import time
from src.EHs.binaryCounterEH import VarEH


def EH_train_epoch(train_loader, network, optimizer, loss_fn, hparams, epoch, taskType='classification'):
    # custom training with resetting of EHs in each epoch.

    # Activate the train=True flag inside the model
    network.train()

    # reset EHs
    network.EHs = [[VarEH(len, eps=network.EHeps, maxValue=1) for len in network.EHlengths] for _ in
                   range(network.numEHs)]

    device = hparams['device']
    avg_loss = None
    avg_weight = 0.1
    acc = 0

    mse = 0

    # For each batch
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = network(data)

        loss = loss_fn(output, target)

        loss.backward()

        if avg_loss:
            avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
        else:
            avg_loss = loss.item()

        if taskType == 'classification':
            # compute number of correct predictions in the batch
            acc += correct_predictions(output, target)

        optimizer.step()

        if batch_idx % hparams['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()), )

    # Average accuracy across all correct predictions batches
    train_acc = 100. * acc / len(train_loader.dataset)
    print('Train accuracy: {:.6f}'.format(train_acc))

    return avg_loss, train_acc


def EH_val_epoch(val_loader, network, hparams, loss_fn):
    # custom training with resetting of EHs in each epoch.

    # Deactivate the train=True flag inside the model
    network.eval()

    # reset EHs
    network.EHs = [[VarEH(len, eps=network.EHeps, maxValue=1) for len in network.EHlengths] for _ in
                   range(network.numEHs)]

    device = hparams['device']
    val_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in val_loader:
            # Load data and feed it through the neural network
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            output = network(data)

            val_loss += loss_fn(output, target, reduction='sum').item()  # sum up batch loss
            # WARNING: If you are using older Torch versions, the previous call may need to be replaced by
            # val_loss += loss_fn(output, target, size_average=False).item()

            # compute number of correct predictions in the batch
            acc += correct_predictions(output, target)

    # Average accuracy across all correct predictions batches now
    val_loss /= len(val_loader.dataset)
    val_acc = 100. * acc / len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, acc, len(val_loader.dataset), val_acc,
    ))
    return val_loss, val_acc


def model_experiment(model, hparams, modelPath, experimentName, seed, train_function,
                     val_function, save_models=False, trainLoader=trainLoaderElec,
                     valLoader=valLoaderElec, testLoader=testLoaderElec):
    "Binary classification"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = model.to(hparams['device'])

    optimizer = optim.RMSprop(model.parameters(), lr=hparams['learning_rate'])
    loss_fn = F.binary_cross_entropy

    print(model)
    print('Num params: ', get_nn_nparams(model))

    # Init lists to save the evolution of the training & tests losses/accuracy.
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_loss = np.inf
    best_val_acc = -np.inf

    total_time = 0

    # For each epoch
    for epoch in range(1, hparams['num_epochs'] + 1):
        startTime = time.time()
        # Compute & save the average training loss for the current epoch
        train_loss = train_function(trainLoader, model, optimizer, loss_fn, hparams, epoch)
        endTime = time.time()
        train_losses.append(train_loss)

        total_time += (endTime - startTime)

        # TIP: Review the functions previously defined to implement the train/tests epochs
        val_loss, val_accuracy = val_function(valLoader, model, hparams, loss_fn)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        # save the model weights
        if val_accuracy > best_val_acc:
            if save_models:
                checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                torch.save(checkpoint, modelPath + experimentName + '.pth')
            best_val_loss = val_loss
            best_val_acc = val_accuracy
            best_model = model

    print("Test accuracy of best model:")

    # use best checkpoint based on val accuracy to compute tests accuracy
    test_loss, test_accuracy = val_function(testLoader, best_model, hparams, loss_fn)

    avg_training_time = total_time / hparams['num_epochs']

    return best_val_acc, test_accuracy, avg_training_time