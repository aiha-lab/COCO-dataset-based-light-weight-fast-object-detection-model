import torch
import torch.nn as nn


class DistillLoss2(nn.Module):

    def __init__(self, channels: int, teacher_channels=None):
        super().__init__()

        self.channels = channels
        if teacher_channels is None:
            teacher_channels = channels
        self.teacher_channels = teacher_channels

        self.l2_loss = nn.MSELoss()

        self.coefficients = {
            "alpha": 4e-4,  # at
            "beta": 2e-2,  # am
            "gamma": 4e-4,  # non-local
            "temperature": 0.5,  # mask
        }

    def forward(self, student: torch.Tensor, teacher: torch.Tensor):
        b, c, h, w = student.shape
        bt, ct, ht, wt = teacher.shape
        assert (b, h, w) == (b, ht, wt) and (c == self.channels) and (ct == self.teacher_channels), \
            f"student: {student.shape}, teacher: {teacher.shape} ({self.channels}, {self.teacher_channels})"

        student_abs = torch.abs(student)
        teacher_abs = torch.abs(teacher)

        at_spatial = torch.mean(student_abs, dim=1, keepdim=True)  # (b, 1, h, w)
        at_spatial_teacher = torch.mean(teacher_abs, dim=1, keepdim=True)
        at_spatial_loss = self.l2_loss(at_spatial, at_spatial_teacher)

        spatial_mask = (at_spatial + at_spatial_teacher).div(self.coefficients["temperature"]).view(b, -1)
        spatial_mask = torch.softmax(spatial_mask, dim=-1).view(b, 1, h, w)
        spatial_mask = spatial_mask * (h * w)

        at_channel = torch.mean(student_abs, dim=[2, 3], keepdim=False)  # (b, c)
        at_channel_teacher = torch.mean(teacher_abs, dim=[2, 3], keepdim=False)
        at_channel_loss = self.l2_loss(at_channel, at_channel_teacher)

        at_loss = at_spatial_loss + at_channel_loss
        at_loss = at_loss * self.coefficients["alpha"]

        channel_mask = (at_channel + at_channel_teacher).div(self.coefficients["temperature"])
        channel_mask = torch.softmax(channel_mask, dim=-1).view(b, c, 1, 1)
        channel_mask = channel_mask * c

        am_loss = torch.sum(torch.square(student - teacher)
                            * spatial_mask * channel_mask).sqrt()
        am_loss = am_loss * self.coefficients["beta"]

        loss = at_loss + am_loss

        return (
            loss,
            at_loss,
            am_loss,
        )


class YOLODistiller2(nn.Module):

    def __init__(self, width=1.0, in_channels=(256, 512, 1024)):
        super().__init__()

        self.dark3 = DistillLoss2(channels=int(in_channels[0] * width))
        self.dark4 = DistillLoss2(channels=int(in_channels[1] * width))
        self.dark5 = DistillLoss2(channels=int(in_channels[2] * width))

        self.C3_p4 = DistillLoss2(channels=int(in_channels[1] * width))
        self.C3_p3 = DistillLoss2(channels=int(in_channels[0] * width))
        self.C3_n3 = DistillLoss2(channels=int(in_channels[1] * width))
        self.C3_n4 = DistillLoss2(channels=int(in_channels[2] * width))

        self.coefficients = {
            "backbone": 1.0,
            "fpn": 1.0,
        }

    def forward(self, student, teacher):
        # order: (dark3, dark4, dark5, C3_p4, C3_p3, C3_n3, C3_n4)

        dark3_loss, _, _ = self.dark3(student[0], teacher[0])
        dark4_loss, _, _ = self.dark4(student[1], teacher[1])
        dark5_loss, _, _ = self.dark5(student[2], teacher[2])
        C3_p4_loss, _, _ = self.C3_p4(student[3], teacher[3])  # noqa
        C3_p3_loss, _, _ = self.C3_p3(student[4], teacher[4])  # noqa
        C3_n3_loss, _, _ = self.C3_n3(student[5], teacher[5])  # noqa
        C3_n4_loss, _, _ = self.C3_n4(student[6], teacher[6])  # noqa

        backbone_loss = (dark3_loss + dark4_loss + dark5_loss) * self.coefficients["backbone"]
        fpn_loss = (C3_p4_loss + C3_p3_loss + C3_n3_loss + C3_n4_loss) * self.coefficients["fpn"]
        loss = backbone_loss + fpn_loss

        return {
            "dis_loss": loss,
            "dis_backbone_loss": backbone_loss,
            "dis_fpn_loss": fpn_loss,
        }
