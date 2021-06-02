from main import *


class Encoder_Gen(nn.Module):

    def __init__(self):
        """
        init neural network with 2 conv and 2 fc.
        :param conv_layers: convolution layers of the second convolution
        :param fc_layers: count fully connected neuron.
        """
        # self.fc_layers = fc_layers
        # self.conv_layers = conv_layers
        super(Encoder_Gen, self).__init__()
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
        """
        :param x:
        :return:
        """
        # do the LNN on data x with relu
        x = self.main(x)
        x = x.view(-1, 256 * 4 * 4)
        x = self.ln1(x)
        x = x.reshape(128, 10, 1, 1)
        return x


class Encoder(nn.Module):
    """
    An Encoder for the mnist dataset
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 10, 7)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """
    A decoder for the mnist dataset
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    """
    A Class that represents an AutoEncoder Network
    """

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def set_loss(self, loss_function):
        self.loss_function = loss_function

    def save_auto_encoder(self, name1, name2):
        encoder_path = name1 + "_encoder.pth"
        decoder_path = name2 + "_decoder.pth"
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)


def train_ae(model, num_epochs=5, batch_size=64, learning_rate=1e-3):
    """
    A train function for the Auto encoder
    :param model:
    :param num_epochs:
    :param batch_size:
    :param learning_rate:
    :return:
    """
    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)  # <--

    train_loader = mnist_data()[1]

    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))
        outputs.append((epoch, img, recon), )
    return outputs


def question4():
    """
    Code for the practical question 4
    :return:
    """
    dataloader = mnist_data()[1]
    saved_encoder_1 = Encoder_Gen().to(device)
    saved_encoder_1.load_state_dict(torch.load("data/encoder.pth", map_location=device))
    saved_decoder_1 = Generator(1, 10, 32, 1).to(device)
    saved_decoder_1.load_state_dict(torch.load("mnist_gan_1gen.pth", map_location=device))
    saved_encoder_2 = Encoder().to(device)
    saved_encoder_2.load_state_dict(torch.load("mnist1_encoder.pth", map_location=device))
    saved_decoder_2 = Decoder().to(device)
    saved_decoder_2.load_state_dict(torch.load("mnist1_decoder.pth", map_location=device))
    img, _ = next(iter(dataloader))
    plt.imshow(img[0].reshape(28, 28), cmap=plt.cm.gray)
    plt.show()
    plt.imshow(img[1].reshape(28, 28), cmap=plt.cm.gray)
    plt.show()
    first_image = img[0].reshape(1, 1, 28, 28)
    second_image = img[1].reshape(1, 1, 28, 28)
    z_one_1 = saved_encoder_1(img)
    z_one_2 = saved_encoder_2(first_image)
    z_two_2 = saved_encoder_2(second_image)
    a_s = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    fig, axs = plt.subplots(2, 11)
    i = 0
    # fig.suptitle('Transition of two numbers using AE')
    for a in a_s:
        output = saved_decoder_2(z_one_2 * a + (1 - a) * z_two_2)
        output = output.detach().cpu()
        if i == 5:
            axs[0, i].set_title('Transition of two numbers using AE')
        axs[0, i].imshow(output.reshape(28, 28), cmap=plt.cm.gray)
        axs[0, i].axis("off")

        i += 1

    i = 0
    for a in a_s:
        picture = z_one_1[0] * a + (1 - a) * z_one_1[1]
        picture = picture.reshape(1, 10, 1, 1)
        output = saved_decoder_1(picture)
        output = output.detach().cpu()
        if i == 5:
            axs[1, i].set_title('Transition of two numbers using Generator')
        axs[1, i].imshow(output.reshape(28, 28), cmap=plt.cm.gray)
        axs[1, i].axis("off")

        i += 1
    plt.show()


class VEncoder(nn.Module):
    def __init__(self):
        super(VEncoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            # nn.Conv2d(32, 10, 7)
        )
        self.mu = nn.Linear(1568, 10)
        self.logvar = nn.Linear(1568, 10)

    def forward(self, x):
        batch = x.shape[0]
        x = self.encoder(x)
        x = x.view(-1, 32 * 7 * 7)
        mu = self.mu(x)
        log = self.logvar(x)
        z = self.reparameterize(mu, log)
        z = z.reshape(batch, 10, 1, 1)
        return z

    def reparameterize(self, mu, log_var):
        # std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(log_var).to(device)  # `randn_like` as we need the same size
        sample = mu + (eps * log_var)  # sampling
        return sample


class VDecoder(nn.Module):
    def __init__(self):
        super(VDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 32, 7),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


def loss_vae(latent_z):
    """
    This function creates the loss for the VAE That helps the z vector to
    be distributed normally
    :param latent_z:
    :return:
    """
    mu = torch.mean(latent_z)
    diff = latent_z - mu
    var = torch.mean(torch.pow(diff, 2.0))
    std = torch.pow(var, 0.5)
    z_s = diff / std
    kurtos = torch.mean(torch.pow(z_s, 4.0))
    return torch.pow(mu, 2) + torch.pow(var - 1, 2) + torch.pow(kurtos - 3, 2)


def train_vae(encoder, decoder, num_of_epochs, criterion, optimizerE, optimizerD, data_loader):
    """
    Train the VAE Network
    :param encoder:
    :param decoder:
    :param num_of_epochs:
    :param criterion:
    :param optimizerE:
    :param optimizerD:
    :param data_loader:
    :return:
    """
    outputs = []
    for i in range(num_of_epochs):
        running_loss = 0.0
        counter = 0
        for data in data_loader:
            # print(counter,flush=True)
            if counter == 468:
                print("hi")
            encoder.zero_grad()
            decoder.zero_grad()
            counter += 1
            img, _ = data
            img = img.to(device)
            latent_z = encoder(img)
            err_z = loss_vae(latent_z) * 7
            output = decoder(latent_z)
            err_mse = criterion(output, img)
            err = err_z + err_mse
            err.backward()
            optimizerE.step()
            optimizerD.step()
            running_loss += err.item()

        print('Epoch:{}, Loss:{:.4f}'.format(i + 1, float(running_loss / counter)))
        outputs.append((i, img, output), )

    return outputs


def save_vae(encoder, decoder, name):
    path_encoder = name + "_encoder_vae_model.pth"
    path_decoder = name + "decoder_vae_model.pth"
    torch.save(encoder.state_dict(), path_encoder)
    torch.save(decoder.state_dict(), path_decoder)


def question5():
    """
    Code for question 5
    :return:
    """
    lr = 0.001
    # epochs = 100
    encoder = VEncoder().to(device)
    decoder = VDecoder().to(device)
    optimizerE = optim.Adam(encoder.parameters(), lr=lr)
    optimizerD = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataloader = mnist_data()[1]
    train_vae(encoder, decoder, 10, criterion, optimizerE, optimizerD, dataloader)
    z_rand = torch.randn(128, 10, 1, 1, device=device)
    fake_images = decoder(z_rand)
    for i in range(20):
        plt.imshow(fake_images[i].cpu().detach().reshape(28, 28), cmap=plt.cm.gray)
        plt.show()


if __name__ == '__main__':
    batch_size = 128
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
