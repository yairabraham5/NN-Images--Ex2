import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch import Tensor
from torch.autograd import Variable
import pandas

PATH = './cifar_net.pth'


def mnist_data():
    """
    Gets the mnist data
    :return:
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    return trainset, dataloader, testset, testloader


def celeb_data():
    """
    Gets the celebA data
    :return:
    """
    image_size = 64
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CelebA(root='./data',
                                           download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)
    return trainset, dataloader


class GeneratorCelebA(nn.Module):
    """
    A Generator for the CelebA dataset
    """

    def __init__(self, ngpu, nz, ngf, nc):
        super(GeneratorCelebA, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class DiscriminatorCelebA(nn.Module):
    """
    Discriminator for the CelebA dataset
    """

    def __init__(self, ngpu, ndf, nc):
        super(DiscriminatorCelebA, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    """
    A Generator for the mnist data set
    """

    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        #  nc = 1

        # # Size of z latent vector (i.e. size of generator input)
        # nz = 10

        # # Size of feature maps in generator
        # ngf = 11

        # # Size of feature maps in discriminator
        # ndf = 11

        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nz = 10
            # ngf = 32
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4 =  2 x 2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Encoder(nn.Module):
    """
    An Encoder for the GAN generator
    """

    def __init__(self):
        """

        """
        # self.fc_layers = fc_layers
        # self.conv_layers = conv_layers
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(256),

        )
        self.ln1 = nn.Linear(256 * 4 * 4, 40)

    def forward(self, x):
        # do the LNN on data x with relu
        x = self.main(x)
        x = x.view(-1, 256 * 4 * 4)
        x = self.ln1(x)
        x = x.reshape(128, 10, 1, 1)
        return x

    def set_loss(self, loss):
        """
        The encoder loss function
        :param loss:
        :return:
        """
        self.loss = loss


class SaturatedBCELoss(nn.BCELoss):
    """
    The Original Loss function that we
    created for the first question
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return -super(SaturatedBCELoss, self).forward(input, 1 - target)


class Discriminator(nn.Module):
    """
    Discriminator for the mnist data set
    """

    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64 = 1 x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32 = ndf x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16 = ndf x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class GAN():
    """
    A GAN Network that has a Generator and a Discriminator
    for mnist or celebA data set
    """

    def __init__(self, generator, discriminator, device, lr, beta1, is_mnist):

        # Learning rate for optimizers
        self.lr = lr
        #
        # # Beta1 hyperparam for Adam optimizers
        self.beta1 = beta1

        # Number of GPUs available. Use 0 for CPU mode.
        ngpu = 1
        self.gen = generator
        self.discriminator = discriminator
        self.loss_function = None
        self.loss_g = []
        self.loss_d = []
        self.images = []
        self.name = ""
        self.real_label = 1.
        self.fake_label = 0.
        if is_mnist:
            self.mnist_init()
        else:
            self.celeb_init()

    def mnist_init(self):
        # mnist data_set
        trainset, self.dataloader, testset, self.testloader = mnist_data()

        # all variables
        nc = 1

        # Size of z latent vector (i.e. size of generator input)
        self.nz = 10

        # Size of feature maps in generator
        ngf = 32

        # Size of feature maps in discriminator
        ndf = 32

        # generator and discriminator
        self.netD = self.discriminator(ngpu, ndf, nc).to(device)
        self.netG = self.gen(ngpu, self.nz, ngf, nc).to(device)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=device)

    def celeb_init(self):
        # init celeb data
        trainset, self.dataloader = celeb_data()

        nc = 3

        # Size of z latent vector (i.e. size of generator input)
        self.nz = 64

        # Size of feature maps in generator
        ngf = 64

        # Size of feature maps in discriminator
        ndf = 64

        self.netD = self.discriminator(ngpu, ndf, nc).to(device)
        self.netG = self.gen(ngpu, self.nz, ngf, nc).to(device)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=device)

    def train_gan(self, num_epochs):
        # Train GAN
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1, 1).squeeze(1)
                # Calculate loss on all-real batch
                errD_real = self.loss_function(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1, 1).squeeze(1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.loss_function(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1, 1).squeeze(1)
                # Calculate G's loss based on this output
                errG = self.loss_function(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(self.dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                self.loss_g.append(errG.item())
                self.loss_d.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    self.images.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

    def set_loss_function(self, lossFunction):
        self.loss_function = lossFunction
        self.name = str(lossFunction)

    def show_error_rate(self):
        plt.plot(self.loss_d, label="discriminator loss")
        plt.plot(self.loss_g, label="generator loss")
        plt.legend()
        plt.show()

    def show_train_result(self):
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in self.images]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())
        plt.show()

    def save_GAN(self, name):
        gen_path = "data/" + name + "gen" + ".pth"
        discriminator_path = "data/" + name + "dis" + ".pth"
        torch.save(self.gen, gen_path)
        torch.save(self.discriminator, discriminator_path)


def save_encoder(encoder):
    """
    A function that saves an encoder network
    :param encoder:
    :return:
    """
    encoder_path = "data/encoder.pth"
    torch.save(encoder.state_dict(), encoder_path)


def train_encoder(encoder, saved_generator, num_of_epochs, optimizer):
    """
    A function that trains an encoder using a pretrained generator
    :param encoder:
    :param saved_generator:
    :param num_of_epochs:
    :param optimizer:
    :return:
    """
    train_loss = []
    for epoch in range(num_of_epochs):
        running_loss = 0.0
        for i in range(200):
            # for data in trainloader:
            # img, _ = data
            # img = img.to(device)
            # img = img.view(img.size(0), -1)
            encoder.zero_grad()
            # optimizer.zero_grad()
            random_vectors = torch.randn(batch_size, 10, 1, 1, device=device)
            images = saved_generator(random_vectors)
            outputs = encoder(images)
            loss = encoder.loss(outputs, random_vectors)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss = running_loss / batch_size
            train_loss.append(loss)

        if epoch % 5 == 0:
            print('Epoch {} of {}, Train Loss: {:.3f}'.format(
                epoch + 1, num_of_epochs, loss))
            #     save_decoded_image(outputs.cpu().data, epoch)
    save_encoder(encoder)
    return train_loss


def gaussNoise(images):
    """
    A function that adds gaussian noise to a batch of images
    :param images:
    :return:
    """
    for i in range(images.shape[0]):
        current_image = images[i]
        current_image = current_image.reshape(28, 28)
        mean = 0
        var = 0.1
        sigma = var * 0.5
        gauss = np.random.normal(mean, sigma, (28, 28))
        current_image = Tensor(gauss.reshape(28, 28)) + current_image
        current_image = current_image.reshape(1, 28, 28)
        images[i] = current_image
    return images


def saltAndPapperNoise(images):
    """
    A function that adds salt and pepper noise to a batch of images
    :param images:
    :return:
    """
    for i in range(images.shape[0]):
        current_image = images[i]
        current_image = current_image.reshape(28, 28)
        s_vs_p = 0.5
        amount = 0.004
        # Salt mode
        num_salt = np.ceil(amount * current_image.shape[0] ** 2 * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in current_image.shape]
        current_image[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * current_image.shape[0] ** 2 * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in current_image.shape]
        current_image[coords] = 0
        current_image = current_image.reshape(1, 28, 28)
        images[i] = current_image
    return images


def poissonNoise(images):
    """
    A function that adds poisson noise to a batch of images
    :param images:
    :return:
    """
    for i in range(images.shape[0]):
        current_image = images[i]
        current_image = current_image.reshape(28, 28)
        mean = 0
        var = 0.3
        sigma = var * 0.5
        gauss = np.random.normal(mean, sigma, (28, 28))
        current_image = Tensor(gauss.reshape(28, 28)) + current_image
        current_image = current_image.reshape(1, 28, 28)
        images[i] = current_image
    return images


def black_stripes_noise(images):
    """
    A function that adds black stripes to the batch of images
    :param images:
    :return:
    """
    for i in range(images.shape[0]):
        current_image = images[i]
        current_image = current_image.reshape(28, 28)
        for i in range(28):
            for j in range(28):
                if i % 2 == 0:
                    current_image[i][j] = 0
        current_image = current_image.reshape(1, 28, 28)
        images[i] = current_image
    return images


def black_stripes_noise_j(images):
    """
    Adds Black stripes to a batch of images
    :param images:
    :return:
    """
    for i in range(images.shape[0]):
        current_image = images[i]
        current_image = current_image.reshape(28, 28)
        for i in range(28):
            for j in range(28):
                if j % 4 == 0:
                    current_image[i][j] = 255
        current_image = current_image.reshape(1, 28, 28)
        images[i] = current_image
    return images


def get_z_from_im(im, gen, lr, maxEpochs, alpha=1e-6):
    """
    This function gets the z vector from a batch of images by iterating again
    and again over the batch of images using a generator
    :param im:
    :param gen:
    :param lr:
    :param maxEpochs:
    :param alpha:
    :return:
    """
    # generator in eval mode
    gen.eval()

    # save the "original" images
    # save_image(x.data, join(exDir, 'original_batch' + str(batchNo) + '.png'), normalize=True, nrow=10)

    # Assume the prior is Standard Normal
    pdf = torch.distributions.Normal(0, 1)

    Zinit = Variable(torch.randn(im.size(0), 64, 1, 1, device=device), requires_grad=True)
    optZ = torch.optim.RMSprop([Zinit], lr=lr)

    for i in range(maxEpochs):
        xHAT = gen.forward(Zinit)
        recLoss = F.mse_loss(xHAT, im)

        # loss to make sure z's are Guassian
        logProb = pdf.log_prob(Zinit).mean(dim=1)
        loss = recLoss - (alpha * logProb.mean())
        optZ.zero_grad()
        loss.backward()
        optZ.step()

    output = gen(Zinit)
    show_images(output, im)


def show_images(batch1, batch2):
    """
    This function shows 10 pictures of two batches of generated pictures
    :param batch1: A batch of pictures
    :param batch2: A batch of pictures
    :return:
    """
    ims1 = [np.transpose(i, (1, 2, 0)) for i in batch1.cpu().detach().numpy()]
    ims2 = [np.transpose(i, (1, 2, 0)) for i in batch2.cpu().detach().numpy()]
    for i in range(10):
        print(i)
        plt.imshow(ims1[i])
        plt.show()
        plt.imshow(ims2[i])
        plt.show()


def question3():
    """
    Code for question 3 of the Pratical Part,
    Adding noise to images and seeing how the encoder and generator handle

    :return:
    """

    dataloader = mnist_data()[1]
    im, _ = next(iter(dataloader))
    # load encoder and generator
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("data/encoder.pth", map_location=device))

    gen = Generator(1, 10, 32, 1).to(device)
    gen.load_state_dict(torch.load("mnist_gan_1gen.pth", map_location=device))

    # Gauss noise
    gauss_im = gaussNoise(im.clone())
    gauss_z = encoder(gauss_im.to(device))
    gen_gauss_image = gen(gauss_z).detach().cpu()

    # Poisson Noise
    poision_im = poissonNoise(im.clone())
    poision_z = encoder(poision_im.to(device))
    gen_poision_image = gen(poision_z).detach().cpu()

    # Salt and pepper noise
    snp_im = saltAndPapperNoise(im.clone())
    snp_z = encoder(snp_im.to(device))
    gen_snp_image = gen(snp_z).detach().cpu()

    # Stripes to the image
    stripes_im = black_stripes_noise(im.clone())
    stripes_z = encoder(stripes_im.to(device))
    gen_stripes_image = gen(stripes_z).detach().cpu()

    for i in range(20):
        fig, axs = plt.subplots(3, 3)
        axs[0, 0].imshow(im[i].reshape(28, 28), cmap=plt.cm.gray)
        axs[0, 0].set_title("Original image", )

        axs[0, 1].imshow(gauss_im[i].reshape(28, 28), cmap=plt.cm.gray)
        axs[0, 1].set_title("Gauss im with noise")
        axs[0, 2].imshow(gen_gauss_image[i].reshape(28, 28), cmap=plt.cm.gray)
        axs[0, 2].set_title("Gauss im with noise after generator")

        axs[1, 0].imshow(snp_im[i].reshape(28, 28), cmap=plt.cm.gray)
        axs[1, 0].set_title("S&P image")
        axs[1, 1].imshow(gen_snp_image[i].reshape(28, 28), cmap=plt.cm.gray)
        axs[1, 1].set_title("S&P image after generator")

        axs[1, 2].imshow(poision_im[i].reshape(28, 28), cmap=plt.cm.gray)
        axs[1, 2].set_title("Poisson image ")
        axs[2, 0].imshow(gen_poision_image[i].reshape(28, 28), cmap=plt.cm.gray)
        axs[2, 0].set_title("Poission image after generator")

        axs[2, 1].imshow(stripes_im[i].reshape(28, 28), cmap=plt.cm.gray)
        axs[2, 1].set_title("Stripes image ")
        axs[2, 2].imshow(gen_stripes_image[i].reshape(28, 28), cmap=plt.cm.gray)
        axs[2, 2].set_title("Stripes image after generator")

        fig.tight_layout()
        plt.show()


def question2():
    """
    Code for question 2 of the Practical part
    :return:
    """
    # load the saved celebA gen
    saved_gen_celeb = GeneratorCelebA(1, 64, 64, 3).to(device)
    saved_gen_celeb.load_state_dict(torch.load("/content/data/Celeba_onegen.pth", map_location=device))
    data, dataloader = celeb_data()
    iter_data = iter(dataloader)
    for i in range(10):
        im, _ = next(iter_data)
        im = im.to(device)
        get_z_from_im(im, saved_gen_celeb, 0.0002, 10)


def question1():
    """
    Code For the first question
    running three different GANS
    :return:
    """
    gan = GAN(Generator, Discriminator, device, 0.0002, 0.5, True)
    gan.set_loss_function(nn.BCELoss())
    gan.train_gan(30)
    gan.show_train_result()
    gan2 = GAN(Generator, Discriminator, device, 0.0002, 0.5, True)
    gan2.set_loss_function(nn.MSELoss())
    gan2.train_gan(30)
    gan2.show_train_result()
    gan3 = GAN(Generator, Discriminator, device, 0.0002, 0.5, True)
    gan3.set_loss_function(SaturatedBCELoss())
    gan3.train_gan(30)
    gan3.show_train_result()


if __name__ == '__main__':
    # Root directory for dataset
    dataroot = "data/celeba"

    # Number of workers for dataloader
    workers = 2
    #
    # # Batch size during training
    batch_size = 128
    # # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Question 2
    # question2()

    # # Question 3
    #
    # load the saved mnist gen and encoder
    saved_gen = Generator(1, 10, 32, 1).to(device)
    saved_gen.load_state_dict(torch.load("mnist_gan_1gen.pth", map_location=device))

    saved_encoder = Encoder().to(device)
    saved_encoder.load_state_dict(torch.load("data/encoder.pth", map_location=device))
    saved_encoder.set_loss(nn.MSELoss())
    # optimizer = optim.Adam(saved_encoder.parameters(), lr=0.0003)
    # train_encoder(saved_encoder, saved_gen, 5, optimizer)
    # save_encoder(saved_encoder)
    # saved_encoder.load_state_dict(torch.load("data/encoder.pth", map_location=device))
