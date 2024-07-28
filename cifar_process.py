import pickle
import cv2


def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


def process_image(img):
    img = img.reshape(3, 32, 32)
    img = img.transpose(1, 2, 0)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def process_set(dataset, name):
    for i in range(dataset[b'data'].shape[0]):
        img = dataset[b'data'][i]
        img = process_image(img)
        path = 'data/MNIST/processed/mnist_test/negative/' + name + str(i) + '.png'
        cv2.imwrite(path, img)
        print(path)


# Load data
train = unpickle('data/cifar-100-python/train')
test = unpickle('data/cifar-100-python/test')

# Process all images to negative dataset
process_set(train, 'cifar100_train')
process_set(test, 'cifar100_test')
