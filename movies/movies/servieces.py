import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm
from util.common import Common

os.environ['KMP_DUPLICATE_LIB_OK']='True'


'''
<참고논문> https://arxiv.org/abs/1511.06434
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
Alec Radford, Luke Metz, Soumith Chintala
In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. 
Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between 
the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional 
generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong 
candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional 
adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. 
Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.
'''


class DcGan(object):

    def __init__(self):
        self.dataroot = "C:/Users/MSJ/AIA/djangoProject/movies/movies/image"
        self.workers = 2
        self.batch_size = 128
        self.image_size = 64
        self.nc = 3
        self.nz = 100
        self.ngf = 64
        self.ndf = 64
        self.num_epochs = 10
        self.lr = 0.0002
        self.beta1 = 0.5
        self.ngpu = 1
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.dataset = dset.ImageFolder(root=self.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(self.image_size),
                                       transforms.CenterCrop(self.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                      shuffle=True, num_workers=self.workers, multiprocessing_context='spawn')
        self.netG = None
        self.netD = None
    def show_image(self):
        # Set random seed for reproducibility
        manualSeed = 999
        manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        dataset = self.dataset
        # Create the dataloader
        dataloader = self.dataloader

        # Decide which device we want to run on
        device = self.device

        # Plot some training images
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

    # custom weights initialization called on netG and netD
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def print_netG(self):
        # Create the generator
        ngpu = self.ngpu
        device = self.device
        netG = Generator(ngpu).to(device)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.02.
        netG.apply(self.weights_init)

        # Print the model
        print(netG)
        self.netG = netG


    def print_netD(self):
        ngpu = self.ngpu
        device = self.device
        # Create the Discriminator
        netD = Discriminator(ngpu).to(device)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netD.apply(self.weights_init)

        # Print the model
        print(netD)
        self.netD = netD

    def generate_face_images(self):
        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(tqdm(self.dataloader)):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.num_epochs - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
            # Output training stats
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, self.num_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # Grab a batch of real images from the dataloader
        real_batch = next(iter(self.dataloader))

        # Plot the real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.show()

    def hook(self):
        self.show_image()
        self.print_netG()
        self.print_netD()
        self.generate_face_images()

    def find_face(self):
        myDlib().zoom()


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        that = DcGan()
        nz = that.nz
        ngf = that.ngf
        nc = that.nc

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

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        that = DcGan()
        ndf = that.ndf
        nc = that.nc

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


import sys
import dlib     # conda isntall -c conda-forge dlib
import cv2
import openface

class myDlib(object):
    def __init__(self):
        pass

    def zoom(self):
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        # 학습된 랜드마크 모델 데이터 경로
        predictor_model = r"C:/Users/MSJ/AIA/djangoProject/movies/movies/files/shape_predictor_68_face_landmarks.dat"

        # HOG 이용한 얼굴 감지 클래스 생성 - dlib
        face_detector = dlib.get_frontal_face_detector()

        # 얼굴에 랜드마크 찾는 클래스 생성 - dlib
        # 매개변수로 랜드마크 모델
        face_pose_predictor = dlib.shape_predictor(predictor_model)

        # 랜드마크를 이용해 얼굴을 정렬할 클래스 생성 - Openface
        # 매개변수로 랜드마크 모델
        face_aligner = openface.AlignDlib(predictor_model)

        # 첫번째 매개변수로 부터 전달 받은 이미지 파일 경로
        file_name = r"C:/Users/MSJ/AIA/djangoProject/movies/movies/files/lenna.jpg"

        # 이미지 파일 경로로 부터 이미지(numpy.ndarry) 불러오기
        image = cv2.imread(file_name)

        '''
         이미지에서 얼굴 찾기
         얼굴 인식 두번째 변수 1은 업샘플링을 한번 하겠다는 얘기인데
         업샘플링을하면 더 많이 인식할 수 있다고 한다.
         다만 값이 커질수록 느리고 메모리도 많이 잡아먹는다.
         그냥 1이면 될 듯. 
        '''
        detected_faces = face_detector(image, 1)

        '''
         detected_faces는 얼굴을 찾은 위치를 가지고 있는 
         "list of rectagles"로 rect형 리스트라고 보면 된다.
         이미지 파일에서 찾은 얼굴 개수 표시 
        '''
        print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

        # 찾은 얼굴 개수 만큼 반복한다.
        for i, face_rect in enumerate(detected_faces):
            '''
            찾은 얼굴 인댁스, 왼쪽, 위, 오른쪽, 아래 위치 (사각형)표시 
            '''
            print(
                "- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                   face_rect.right(),
                                                                                   face_rect.bottom()))

            # 얼굴 위치에서 랜드마크 찾기
            pose_landmarks = face_pose_predictor(image, face_rect)
            '''
            pose_landmarks는 dlib의 full_object_detection 클래스이고 
            num_parts
                랜드마크 개수 - 68이 나와야 정상 

            part(idx) → dlib.point
                idx(랜드마크 번호) point(x, y) 변수

            parts() → dlib.points
                랜드마크 전체의 points 
            rect
                얼굴 위치 left(), top(), right(), bottom() 
            '''

            '''	    
            인식된 랜드마크를 openface를 이용해 변환
            532 - imgDim
                이미지 크기 532는 532x532 이미지로 반환하겠다는 뜻 
            image - rgbImg
                변환 시킬 원본 이미지 : (높이, 너비, 3)
            face_rect - bb
                얼굴 위치 (rect)
            landmarkIndices
                변환 대상의 인덱스.
                openface.AlignDlib.OUTER_EYES_AND_NOSE
                 [36, 45, 33]
                openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
                 [39, 42, 57]
            '''
            alignedFace = face_aligner.align(532, image, face_rect,
                                             landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            '''
            alignedFace 는 RGB(ndarray) 이미지      
            '''

            # aligned_face_x.jpg 로 저장
            cv2.imwrite("../files/aligned_face_{}.jpg".format(i), alignedFace)




MENUS = ["Exit",  #0
        "/movies/movies/fake-faces",  #1. Loading CelebA Dataset
        "/movies/movies/face-blow-up",  # 2. Blow Up Face By DLib
         ]
menus = {
    "1" : lambda d: d.hook(),
    "2" : lambda d: d.find_face()
}
def my_menu(ls):
    for i, j in enumerate(ls):
        print(f"{i}. {j}")
    return input('메뉴선택: ')


if __name__ == '__main__':
    d = DcGan()
    while True:
        menu = my_menu(MENUS)
        if menu == "0":
            print(" ### 종료 ### ")
            break
        else:
            menus[menu](d)