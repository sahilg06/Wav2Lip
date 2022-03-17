import torch
from models import Wav2Lip as Wav2Lip

device = torch.device("cuda")

pretrain_sd = torch.load('/home/mansi/sg_deepfakes/checkpoints_downloaded/wav2lip_gan.pth')
print(len(pretrain_sd['state_dict']))

pretrain_sd_face = {k:v for k,v in pretrain_sd['state_dict'].items() if k.split('.')[0]=='face_encoder_blocks'}
print(len(pretrain_sd_face))
# print(pretrain_sd_face)

model = Wav2Lip().to(device)
model_sd = model.state_dict()

temp_model_sd = model_sd
model_sd.update(pretrain_sd_face)
# print(model_sd == temp_model_sd)
model.load_state_dict(model_sd)

model_sd_face = {k:v for k,v in model.state_dict().items() if k.split('.')[0]=='face_encoder_blocks'}
print(model_sd_face)

# for (k1,v1),(k2,v2) in zip(model.state_dict().items(),r.items()):
#     print(k1,k2)
#     print(v1==v2)
