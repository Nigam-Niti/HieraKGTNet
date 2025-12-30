def superpixel_segmentation(image, flag=0):
  # Global
  if flag==0:
      ld = int(''.join(map(str, image[1,1].shape)))
      xc = img_as_float(image.copy())
      xcp = img_as_float(image.copy())
      for i in range(ld):
        xcp[:,:,i]= slic(xc[:,:,i],n_segments = 300,compactness=10, channel_axis=None)
      return xcp

  # Local
  elif flag==1:
      ld = int(''.join(map(str, image[1,1].shape)))
      xc = img_as_float(image.copy())
      xcp = img_as_float(image.copy())
      for i in range(ld):
        xcp[:,:,i]= slic(xc[:,:,i],n_segments = 500,compactness=1, channel_axis=None)
      return xcp



def kmeansnew(image):
  last_dimension = int(''.join(map(str, image[1,1].shape)))
  pixel_vals = image.reshape((-1,last_dimension))
  # Convert to float type
  pixel_vals = np.float32(pixel_vals)
  # The below line of code defines the criteria for the algorithm to stop running,
  #which will happen is 100 iterations are run, or the epsilon (which is the required accuracy)
  #becomes 85%
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1)

  # then perform k-means clustering with h number of clusters defined as 3
  #also random centres are initially chosen for k-means clustering
  k = output_units
  retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  # convert data into 8-bit values
  centers = np.uint8(centers)
  segmented_data = centers[labels.flatten()]
  # reshape data into the original image dimensions
  segmented_image = segmented_data.reshape((image.shape))
  return segmented_image

def custom_softmax(y, axis=1):
  """Softmax function for 2D array (rows)"""
  for i in range(9):
    e_x = np.exp(y[:,:,i] - np.max(y[:,:,i], axis=axis, keepdims=True))
    y[:,:,i] = e_x / np.sum(e_x, axis=1, keepdims=True)
  return y


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def loadData(name):
    data_path = os.path.join(os.getcwd(), 'drive/MyDrive/LSGA-VIT/data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'Houston':
        data = sio.loadmat(os.path.join(data_path, 'Houston.mat'))['Houston']
        labels = sio.loadmat(os.path.join(data_path, 'Houston_gt.mat'))['gt']
    elif name == "LK":
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
    return data, labels

class TrainDS(torch.utils.data.Dataset):
    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)
    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]
    def __len__(self):

        return self.len

class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]
    def __len__(self):

        return self.len

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)

def kappa(testData, k):
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i] * 1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)

    # Pe = float(np.sum(ysum * xsum)) / (np.sum(dataMat) ** 2)

    Pe = float(np.sum(ysum * xsum)) /(( np.sum(dataMat) ** 2) + 0.000001)
    P0 = float(P0 / (np.sum(dataMat) + 0.000001) * 1.0)
    cohens_coefficient = float((P0 - Pe) / ((1 - Pe)+ 0.000001))
    return cohens_coefficient

def print_and_save(output, filename=f"{Options().datasetname}_{Options().numtrain}.txt"):
    # data_path = os.path.join(os.getcwd(), 'drive/MyDrive/LSGA-VIT/output')
    print(output)  # Print to console
    # with open(os.path.join(data_path, filename), "a") as f:  # Append to file
    #     f.write(output + "\n")
