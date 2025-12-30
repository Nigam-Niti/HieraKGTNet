if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = False

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = opt.datasetname
X, y = loadData(dataset)
H = X.shape[0]
W = X.shape[1]
pca_components = opt.spectrumnum
print('Hyperspectral data shape:', X.shape)
print('Label shape:', y.shape)
sample_number = np.count_nonzero(y)
print('the number of sample:', sample_number)

X_pca_1 = applyPCA(X, numComponents=9)
X_slic_global = superpixel_segmentation(X_pca_1, flag=0)   # global
X_soft_1 = custom_softmax(X_slic_global) + X_slic_global
X_slic_1 = np.concatenate((X_soft_1, X_pca_1), axis = 2)


X_pca_2 = applyPCA(X, numComponents=9)
X_slic_local = superpixel_segmentation(X_pca_1, flag=1)   # local
X_soft_2 = custom_softmax(X_slic_local) + X_slic_local
X_slic_2 = np.concatenate((X_soft_2, X_pca_2), axis = 2)

X_pca = np.concatenate((X_slic_1, X_slic_2), axis = 2)
# X_pca.shape

print(f"X_slic_1: {str(X_slic_1.shape)}")
print(f"X_slic_2: {str(X_slic_2.shape)}")
print(f"X_pca: {str(X_pca.shape)}")

[nRow, nColumn, nBand] = X_pca.shape
num_class = int(np.max(y))
windowsize = opt.windowsize
Wid = opt.inputsize
halfsizeTL = int((Wid-1)/2)
halfsizeBR = int((Wid-1)/2)
paddedDatax = cv2.copyMakeBorder(X_pca, halfsizeTL, halfsizeBR, halfsizeTL, halfsizeBR, cv2.BORDER_CONSTANT, 0)  #cv2.BORDER_REPLICAT周围值
paddedDatay = cv2.copyMakeBorder(y, halfsizeTL, halfsizeBR, halfsizeTL, halfsizeBR, cv2.BORDER_CONSTANT, 0)
patchIndex = 0
X_patch = np.zeros((sample_number, Wid, Wid, pca_components))
y_patch = np.zeros(sample_number)
for h in range(0, paddedDatax.shape[0]):
    for w in range(0, paddedDatax.shape[1]):
        if paddedDatay[h, w] == 0:
            continue
        X_patch[patchIndex, :, :, :] = paddedDatax[h-halfsizeTL:h+halfsizeBR+1, w-halfsizeTL:w+halfsizeBR+1, :]
        X_patch[patchIndex] = paddedDatay[h, w]
        patchIndex = patchIndex + 1


X_train_p = patchify(paddedDatax, (Wid, Wid, pca_components), step=1)
if opt.input3D:
    X_train_p = X_train_p.reshape(-1, Wid, Wid, pca_components, 1)
else:
    X_train_p = X_train_p.reshape(-1, Wid, Wid, pca_components)
y_train_p = y.reshape(-1)
indices_0 = np.arange(y_train_p.size)
X_train_q = X_train_p[y_train_p > 0, :, :, :]
y_train_q = y_train_p[y_train_p > 0]
indices_1 = indices_0[y_train_p > 0]
y_train_q -= 1
X_train_q = X_train_q.transpose(0, 3, 1, 2)
Xtrain, Xtest, ytrain, ytest, idx1, idx2 = train_test_split(X_train_q, y_train_q, indices_1,
                                                            train_size=opt.numtrain,
                                                            stratify=y_train_q)
print('after Xtrain shape:', Xtrain.shape)
print('after Xtest shape:', Xtest.shape)

trainset = TrainDS(Xtrain, ytrain)
testset = TestDS(Xtest, ytest)
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=opt.batchSize, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=opt.batchSize, shuffle=False, num_workers=0)
nz = int(opt.nz)
nc = pca_components
output_units = num_class
nb_label = num_class
print("label", nb_label)

def train(netD, train_loader, test_loader):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    flop_counts = []

    for epoch in range(1, opt.epochs + 1):
        netD.train()
        right = 0
        for i, datas in enumerate(train_loader):
            netD.zero_grad()
            img, label = datas
            batch_size = img.size(0)
            input.resize_(img.size()).copy_(img)
            c_label.resize_(batch_size).copy_(label)
            c_output = netD(input)
            c_errD_real = c_criterion(c_output, c_label)
            errD_real = c_errD_real
            errD_real.backward()
            D_x = c_output.data.mean()
            correct, length = test(c_output, c_label)

            input_size = img.numel()
            output_size = c_output.numel()
            flops = input_size * output_size  # placeholder
            flop_counts.append(flops)

            optimizerD.step()
            right += correct

        if epoch % 5 == 0:
            output_str = '[%d/%d][%d/%d]   D(x): %.4f, errD_real: %.4f,  Accuracy: %.4f / %.4f = %.4f' % (
                epoch, opt.epochs, i, len(train_loader),
                D_x, errD_real,
                right, len(train_loader.dataset), 100. * right / len(train_loader.dataset)
            )
            print_and_save(output_str)  # Save to both console and file

        if epoch % 5 == 0:
            netD.eval()
            test_loss = 0
            right = 0
            all_Label = []
            all_target = []
            for data, target in test_loader:
                indx_target = target.clone()
                if opt.cuda:
                    data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    data, target = Variable(data), Variable(target)

                start.record(stream=torch.cuda.current_stream())
                output = netD(data)
                end.record(stream=torch.cuda.current_stream())
                end.synchronize()
                test_loss += c_criterion(output, target).item()
                pred = output.max(1)[1]  # get the index of the max log-probability
                all_Label.extend(pred)
                all_target.extend(target)
                right += pred.cpu().eq(indx_target).sum()

            test_loss = test_loss / len(test_loader)  # average over number of mini-batch
            acc = float(100. * float(right)) / float(len(test_loader.dataset))
            test_str = '\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, right, len(test_loader.dataset), acc)
            print_and_save(test_str)  # Save to both console and file

            AAA = torch.stack(all_target).data.cpu().numpy()
            BBB = torch.stack(all_Label).data.cpu().numpy()
            C = confusion_matrix(AAA, BBB)
            C = C[:num_class, :num_class]
            k = kappa(C, np.shape(C)[0])
            AA_ACC = np.diag(C) / np.sum(C, 1)
            AA = np.mean(AA_ACC, 0)

            if math.isnan(acc):
                acc = 0

            result_str = 'OA= %.5f AA= %.5f k= %.5f' % (acc, AA, k)
            print_and_save(result_str)  # Save to both console and file

            avg_flops = sum(flop_counts) / len(flop_counts) if flop_counts else 0
            flop_str = f"Average FLOPs per iteration: {avg_flops}"
            print_and_save(flop_str)  # Save to both console and file

total_train_time = 0  # Initialize total training time

for index_iter in range(1):
    print_and_save(f'iter: {index_iter}')

    netD = LSGAVIT(img_size=Wid,
                   patch_size=3,
                   in_chans=pca_components,
                   num_classes=num_class,
                   embed_dim=60,
                   depths=[2],
                   num_heads=[12, 12, 12, 24],
                   )
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))

    c_criterion = FocalLoss(gamma=3.75, device="cuda")
    input = torch.FloatTensor(opt.batchSize, nc, opt.inputsize, opt.inputsize)
    c_label = torch.LongTensor(opt.batchSize)
    if opt.cuda:
        netD.cuda()
        c_criterion.cuda()
        input = input.cuda()
        c_label = c_label.cuda()
    input = Variable(input)
    c_label = Variable(c_label)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.D_lr)

    # Start time measurement
    start_time = time.time()

    # Train the model
    train(netD, train_loader, test_loader)

    # End time measurement
    end_time = time.time()

    # Calculate the time for this iteration
    iter_train_time = end_time - start_time
    total_train_time += iter_train_time  # Accumulate training time

    print_and_save(f"Time taken for iteration {index_iter}: {iter_train_time:.2f} seconds")

print_and_save(f"Total training time: {total_train_time:.2f} seconds")
netD.eval()
all_preds = []
all_targets = []
start_time = time.time()
for data, target in test_loader:
    if opt.cuda:
        data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        output = netD(data)
        preds = output.max(1)[1].cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(target.cpu().numpy())
end_time = time.time()

print_and_save(f"Total inference time: {end_time-start_time:.2f} seconds")

# Generate classification report
report = classification_report(all_targets, all_preds)
print_and_save("Classification Report:\n" + report)

# Generate confusion matrix
cm = confusion_matrix(all_targets, all_preds)

# Format confusion matrix with class labels
class_labels = list(range(num_class))
df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)

# Save confusion matrix
print_and_save(f"Confusion Matrix:\n{df_cm}")
