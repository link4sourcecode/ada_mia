import torch
from torchvision import transforms
import torch.nn as nn
import warnings

max, interval = 1, 0.05
warnings.filterwarnings('ignore')

# main purpose -> under weak augmentation, it's difficult to distinguish members from non-members
#              -> under aggressive augmentation, we can seek some differences
#              -> the progressive augmentation help us define what is so-called "aggressive"


# define the function of global DA setting
# DA transformations : ResizeCrop -> R  HorizontalFlip -> H  ColorJitter -> C  Grayscale -> G
# DA strength : Weak -> W  Strong -> S  Aggressive -> A
def data_aug_method(strength=0.1, mode='sw', combination='r+f+c+g'):
    # data_transforms -> type: transforms.Compose([])

    # first under the weak setting, only using one single DA transformation
    if mode == 'W' and combination == 'R' and 0 <= strength <= 0.3:
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0, 1 - strength)),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'W' and combination == 'H' and 0 <= strength <= 0.3:
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                              transforms.RandomHorizontalFlip(p=1),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'W' and combination == 'C' and 0 <= strength <= 0.3:
        color_jitter = transforms.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                              transforms.RandomApply([color_jitter], p=1),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'W' and combination == 'G' and 0 <= strength <= 1:
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                              transforms.RandomGrayscale(p=1),
                                              transforms.ToTensor()
                                              ])

    # second under the strong setting, using two DA transformations
    elif mode == 'S' and combination == 'R+H' and 0.3 <= strength <= 0.6:
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0, 1 - strength)),
                                              transforms.RandomHorizontalFlip(p=1),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'S' and combination == 'R+C' and 0.3 <= strength <= 0.6:
        color_jitter = transforms.ColorJitter(1.6 * strength, 1.6 * strength, 1.6 * strength, 0.4 * strength)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0, 1 - strength)),
                                              transforms.RandomApply([color_jitter], p=1),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'S' and combination == 'R+G' and 0.3 <= strength <= 0.6:
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0, 1 - strength)),
                                              transforms.RandomGrayscale(p=1),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'S' and combination == 'H+C' and 0.3 <= strength <= 0.6:
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                              transforms.RandomHorizontalFlip(p=1),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'S' and combination == 'H+G' and 0.3 <= strength <= 0.6:
        color_jitter = transforms.ColorJitter(1.6 * strength, 1.6 * strength, 1.6 * strength, 0.4 * strength)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                              transforms.RandomApply([color_jitter], p=1),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'S' and combination == 'C+G' and 0.3 <= strength <= 0.6:
        color_jitter = transforms.ColorJitter(1.6 * strength, 1.6 * strength, 1.6 * strength, 0.4 * strength)
        data_transforms = transforms.Compose([transforms.RandomApply([color_jitter], p=1),
                                              transforms.RandomGrayscale(p=1),
                                              transforms.ToTensor()
                                              ])

    # third under the aggressive DA, using third or more DA transformations
    elif mode == 'A' and combination == 'R+H+C' and 0.6 <= strength <= 1:
        color_jitter = transforms.ColorJitter(1.6 * strength, 1.6 * strength, 1.6 * strength, 0.4 * strength)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0, 1 - strength)),
                                              transforms.RandomHorizontalFlip(p=1),
                                              transforms.RandomApply([color_jitter], p=1),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'A' and combination == 'R+H+G' and 0.6 <= strength <= 1:
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0, 1 - strength)),
                                              transforms.RandomHorizontalFlip(p=1),
                                              transforms.RandomGrayscale(p=1),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'A' and combination == 'R+C+G' and 0.6 <= strength <= 1:
        color_jitter = transforms.ColorJitter(1.6 * strength, 1.6 * strength, 1.6 * strength, 0.4 * strength)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0, 1 - strength)),
                                              transforms.RandomApply([color_jitter], p=1),
                                              transforms.RandomGrayscale(p=1),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'A' and combination == 'H+C+G' and 0.6 <= strength <= 1:
        color_jitter = transforms.ColorJitter(1.6 * strength, 1.6 * strength, 1.6 * strength, 0.4 * strength)
        data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                              transforms.RandomApply([color_jitter], p=1),
                                              transforms.RandomGrayscale(p=1),
                                              transforms.ToTensor()
                                              ])
    elif mode == 'A' and combination == 'R+H+C+G' and 0.6 <= strength <= 1:
        color_jitter = transforms.ColorJitter(1.6 * strength, 1.6 * strength, 1.6 * strength, 0.4 * strength)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0, 1 - strength)),
                                              transforms.RandomHorizontalFlip(p=1),
                                              transforms.RandomApply([color_jitter], p=1),
                                              transforms.RandomGrayscale(p=1),
                                              transforms.ToTensor()
                                              ])

    # if setting is illegal, raise error
    else:
        raise KeyboardInterrupt('DA setting is illegal, please check it out!\n')

    return data_transforms


def data_aug_no():
    data_transforms = transforms.Compose([transforms.Resize(32),
                                          transforms.ToTensor(),
                                          ])
    return data_transforms


class ADAMIA_DataTransform(object):
    def __init__(self, transform1, transform2, transform3, transform4, transform5, transform6, transform7,
                 transform8, transform9, transform10, transform11, transform12, transform13, transform14, transform15,
                 transform16, transform17, transform18, transform19, transform20, transform21, transform22,
                 transform23, transform24, transform25, transform26, transform27, transform28, transform29,
                 transform30, transform31, transform32, transform33, transform34, transform35, transform36,
                 transform37, transform38, transform39, transform40, transform41, transform42, transform43,
                 transform44, transform45, transform46, transform47, transform48, transform49, transform50,
                 transform51, transform52, transform53, transform54, transform55, transform56, transform57,
                 transform58, transform59, transform60, transform61, transform62, transform63, transform64,
                 transform65, transform66, transform67, transform68, transform69, transform70, transform71,
                 transform72, transform73, transform74, transform75, transform76, transform77, transform78,
                 transform79, transform80, transform81, transform82, transform83, transform84, transform85,
                 transform86, transform87, transform88, transform89, transform90, transform91, transform92,
                 transform93, transform94, transform95
                 ):
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4
        self.transform5 = transform5
        self.transform6 = transform6
        self.transform7 = transform7
        self.transform8 = transform8
        self.transform9 = transform9
        self.transform10 = transform10
        self.transform11 = transform11
        self.transform12 = transform12
        self.transform13 = transform13
        self.transform14 = transform14
        self.transform15 = transform15
        self.transform16 = transform16
        self.transform17 = transform17
        self.transform18 = transform18
        self.transform19 = transform19
        self.transform20 = transform20
        self.transform21 = transform21
        self.transform22 = transform22
        self.transform23 = transform23
        self.transform24 = transform24
        self.transform25 = transform25
        self.transform26 = transform26
        self.transform27 = transform27
        self.transform28 = transform28
        self.transform29 = transform29
        self.transform30 = transform30
        self.transform31 = transform31
        self.transform32 = transform32
        self.transform33 = transform33
        self.transform34 = transform34
        self.transform35 = transform35
        self.transform36 = transform36
        self.transform37 = transform37
        self.transform38 = transform38
        self.transform39 = transform39
        self.transform40 = transform40
        self.transform41 = transform41
        self.transform42 = transform42
        self.transform43 = transform43
        self.transform44 = transform44
        self.transform45 = transform45
        self.transform46 = transform46
        self.transform47 = transform47
        self.transform48 = transform48
        self.transform49 = transform49
        self.transform50 = transform50
        self.transform51 = transform51
        self.transform52 = transform52
        self.transform53 = transform53
        self.transform54 = transform54
        self.transform55 = transform55
        self.transform56 = transform56
        self.transform57 = transform57
        self.transform58 = transform58
        self.transform59 = transform59
        self.transform60 = transform60
        self.transform61 = transform61
        self.transform62 = transform62
        self.transform63 = transform63
        self.transform64 = transform64
        self.transform65 = transform65
        self.transform66 = transform66
        self.transform67 = transform67
        self.transform68 = transform68
        self.transform69 = transform69
        self.transform70 = transform70
        self.transform71 = transform71
        self.transform72 = transform72
        self.transform73 = transform73
        self.transform74 = transform74
        self.transform75 = transform75
        self.transform76 = transform76
        self.transform77 = transform77
        self.transform78 = transform78
        self.transform79 = transform79
        self.transform80 = transform80
        self.transform81 = transform81
        self.transform82 = transform82
        self.transform83 = transform83
        self.transform84 = transform84
        self.transform85 = transform85
        self.transform86 = transform86
        self.transform87 = transform87
        self.transform88 = transform88
        self.transform89 = transform89
        self.transform90 = transform90
        self.transform91 = transform91
        self.transform92 = transform92
        self.transform93 = transform93
        self.transform94 = transform94
        self.transform95 = transform95

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        x3 = self.transform3(sample)
        x4 = self.transform4(sample)
        x5 = self.transform5(sample)
        x6 = self.transform6(sample)
        x7 = self.transform7(sample)
        x8 = self.transform8(sample)
        x9 = self.transform9(sample)
        x10 = self.transform10(sample)
        x11 = self.transform11(sample)
        x12 = self.transform12(sample)
        x13 = self.transform13(sample)
        x14 = self.transform14(sample)
        x15 = self.transform15(sample)
        x16 = self.transform16(sample)
        x17 = self.transform17(sample)
        x18 = self.transform18(sample)
        x19 = self.transform19(sample)
        x20 = self.transform20(sample)
        x21 = self.transform21(sample)
        x22 = self.transform22(sample)
        x23 = self.transform23(sample)
        x24 = self.transform24(sample)
        x25 = self.transform25(sample)
        x26 = self.transform26(sample)
        x27 = self.transform27(sample)
        x28 = self.transform28(sample)
        x29 = self.transform29(sample)
        x30 = self.transform30(sample)
        x31 = self.transform31(sample)
        x32 = self.transform32(sample)
        x33 = self.transform33(sample)
        x34 = self.transform34(sample)
        x35 = self.transform35(sample)
        x36 = self.transform36(sample)
        x37 = self.transform37(sample)
        x38 = self.transform38(sample)
        x39 = self.transform39(sample)
        x40 = self.transform40(sample)
        x41 = self.transform41(sample)
        x42 = self.transform42(sample)
        x43 = self.transform43(sample)
        x44 = self.transform44(sample)
        x45 = self.transform45(sample)
        x46 = self.transform46(sample)
        x47 = self.transform47(sample)
        x48 = self.transform48(sample)
        x49 = self.transform49(sample)
        x50 = self.transform50(sample)
        x51 = self.transform51(sample)
        x52 = self.transform52(sample)
        x53 = self.transform53(sample)
        x54 = self.transform54(sample)
        x55 = self.transform55(sample)
        x56 = self.transform56(sample)
        x57 = self.transform57(sample)
        x58 = self.transform58(sample)
        x59 = self.transform59(sample)
        x60 = self.transform60(sample)
        x61 = self.transform61(sample)
        x62 = self.transform62(sample)
        x63 = self.transform63(sample)
        x64 = self.transform64(sample)
        x65 = self.transform65(sample)
        x66 = self.transform66(sample)
        x67 = self.transform67(sample)
        x68 = self.transform68(sample)
        x69 = self.transform69(sample)
        x70 = self.transform70(sample)
        x71 = self.transform71(sample)
        x72 = self.transform72(sample)
        x73 = self.transform73(sample)
        x74 = self.transform74(sample)
        x75 = self.transform75(sample)
        x76 = self.transform76(sample)
        x77 = self.transform77(sample)
        x78 = self.transform78(sample)
        x79 = self.transform79(sample)
        x80 = self.transform80(sample)
        x81 = self.transform81(sample)
        x82 = self.transform82(sample)
        x83 = self.transform83(sample)
        x84 = self.transform84(sample)
        x85 = self.transform85(sample)
        x86 = self.transform86(sample)
        x87 = self.transform87(sample)
        x88 = self.transform88(sample)
        x89 = self.transform89(sample)
        x90 = self.transform90(sample)
        x91 = self.transform91(sample)
        x92 = self.transform92(sample)
        x93 = self.transform93(sample)
        x94 = self.transform94(sample)
        x95 = self.transform95(sample)

        return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, \
            x23, x24, x25, x26, x27, x28,x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, \
            x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64,\
            x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85, \
            x86, x87, x88, x89, x90, x91, x92, x93, x94, x95


def trans():
    return ADAMIA_DataTransform(data_aug_no(),
                                # super weak DA setting
                                data_aug_method(strength=0, mode='W', combination='R'),
                                data_aug_method(strength=0.05, mode='W', combination='R'),
                                data_aug_method(strength=0.1, mode='W', combination='R'),
                                data_aug_method(strength=0.15, mode='W', combination='R'),
                                data_aug_method(strength=0.2, mode='W', combination='R'),
                                data_aug_method(strength=0.25, mode='W', combination='R'),
                                data_aug_method(strength=0.3, mode='W', combination='R'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0, mode='W', combination='H'),
                                data_aug_method(strength=0.05, mode='W', combination='H'),
                                data_aug_method(strength=0.1, mode='W', combination='H'),
                                data_aug_method(strength=0.15, mode='W', combination='H'),
                                data_aug_method(strength=0.2, mode='W', combination='H'),
                                data_aug_method(strength=0.25, mode='W', combination='H'),
                                data_aug_method(strength=0.3, mode='W', combination='H'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0, mode='W', combination='C'),
                                data_aug_method(strength=0.05, mode='W', combination='C'),
                                data_aug_method(strength=0.1, mode='W', combination='C'),
                                data_aug_method(strength=0.15, mode='W', combination='C'),
                                data_aug_method(strength=0.2, mode='W', combination='C'),
                                data_aug_method(strength=0.25, mode='W', combination='C'),
                                data_aug_method(strength=0.3, mode='W', combination='C'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0, mode='W', combination='G'),
                                data_aug_method(strength=0.05, mode='W', combination='G'),
                                data_aug_method(strength=0.1, mode='W', combination='G'),
                                data_aug_method(strength=0.15, mode='W', combination='G'),
                                data_aug_method(strength=0.2, mode='W', combination='G'),
                                data_aug_method(strength=0.25, mode='W', combination='G'),
                                data_aug_method(strength=0.3, mode='W', combination='G'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0.4, mode='S', combination='R+H'),
                                data_aug_method(strength=0.45, mode='S', combination='R+H'),
                                data_aug_method(strength=0.5, mode='S', combination='R+H'),
                                data_aug_method(strength=0.55, mode='S', combination='R+H'),
                                data_aug_method(strength=0.6, mode='S', combination='R+H'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0.4, mode='S', combination='R+C'),
                                data_aug_method(strength=0.45, mode='S', combination='R+C'),
                                data_aug_method(strength=0.5, mode='S', combination='R+C'),
                                data_aug_method(strength=0.55, mode='S', combination='R+C'),
                                data_aug_method(strength=0.6, mode='S', combination='R+C'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0.4, mode='S', combination='R+G'),
                                data_aug_method(strength=0.45, mode='S', combination='R+G'),
                                data_aug_method(strength=0.5, mode='S', combination='R+G'),
                                data_aug_method(strength=0.55, mode='S', combination='R+G'),
                                data_aug_method(strength=0.6, mode='S', combination='R+G'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0.4, mode='S', combination='C+G'),
                                data_aug_method(strength=0.45, mode='S', combination='C+G'),
                                data_aug_method(strength=0.5, mode='S', combination='C+G'),
                                data_aug_method(strength=0.55, mode='S', combination='C+G'),
                                data_aug_method(strength=0.6, mode='S', combination='C+G'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0.4, mode='S', combination='H+C'),
                                data_aug_method(strength=0.45, mode='S', combination='H+C'),
                                data_aug_method(strength=0.5, mode='S', combination='H+C'),
                                data_aug_method(strength=0.55, mode='S', combination='H+C'),
                                data_aug_method(strength=0.6, mode='S', combination='H+C'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0.4, mode='S', combination='C+G'),
                                data_aug_method(strength=0.45, mode='S', combination='C+G'),
                                data_aug_method(strength=0.5, mode='S', combination='C+G'),
                                data_aug_method(strength=0.55, mode='S', combination='C+G'),
                                data_aug_method(strength=0.6, mode='S', combination='C+G'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0.6, mode='A', combination='R+H+C'),
                                data_aug_method(strength=0.65, mode='A', combination='R+H+C'),
                                data_aug_method(strength=0.7, mode='A', combination='R+H+C'),
                                data_aug_method(strength=0.75, mode='A', combination='R+H+C'),
                                data_aug_method(strength=0.8, mode='A', combination='R+H+C'),
                                data_aug_method(strength=0.85, mode='A', combination='R+H+C'),
                                data_aug_method(strength=0.9, mode='A', combination='R+H+C'),
                                data_aug_method(strength=0.95, mode='A', combination='R+H+C'),
                                data_aug_method(strength=1, mode='A', combination='R+H+C'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0.6, mode='A', combination='R+H+G'),
                                data_aug_method(strength=0.65, mode='A', combination='R+H+G'),
                                data_aug_method(strength=0.7, mode='A', combination='R+H+G'),
                                data_aug_method(strength=0.75, mode='A', combination='R+H+G'),
                                data_aug_method(strength=0.8, mode='A', combination='R+H+G'),
                                data_aug_method(strength=0.85, mode='A', combination='R+H+G'),
                                data_aug_method(strength=0.9, mode='A', combination='R+H+G'),
                                data_aug_method(strength=0.95, mode='A', combination='R+H+G'),
                                data_aug_method(strength=1, mode='A', combination='R+H+G'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0.6, mode='A', combination='R+C+G'),
                                data_aug_method(strength=0.65, mode='A', combination='R+C+G'),
                                data_aug_method(strength=0.7, mode='A', combination='R+C+G'),
                                data_aug_method(strength=0.75, mode='A', combination='R+C+G'),
                                data_aug_method(strength=0.8, mode='A', combination='R+C+G'),
                                data_aug_method(strength=0.85, mode='A', combination='R+C+G'),
                                data_aug_method(strength=0.9, mode='A', combination='R+C+G'),
                                data_aug_method(strength=0.95, mode='A', combination='R+C+G'),
                                data_aug_method(strength=1, mode='A', combination='R+C+G'),
                                # ----------------------------------------------------------
                                data_aug_method(strength=0.6, mode='A', combination='R+H+C+G'),
                                data_aug_method(strength=0.65, mode='A', combination='R+H+C+G'),
                                data_aug_method(strength=0.7, mode='A', combination='R+H+C+G'),
                                data_aug_method(strength=0.75, mode='A', combination='R+H+C+G'),
                                data_aug_method(strength=0.8, mode='A', combination='R+H+C+G'),
                                data_aug_method(strength=0.85, mode='A', combination='R+H+C+G'),
                                data_aug_method(strength=0.9, mode='A', combination='R+H+C+G'),
                                data_aug_method(strength=0.95, mode='A', combination='R+H+C+G'),
                                data_aug_method(strength=1, mode='A', combination='R+H+C+G'),
                                )





