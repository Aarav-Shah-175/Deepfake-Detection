import torch
from model import DeepfakeStudent
from torch.utils.mobile_optimizer import optimize_for_mobile

def export_torchscript(weights='models/best_student.pt',out='exports/student_traced.pt',frames=1,apply_mobile_opt=True,apply_dynamic_quant=False):
 model=DeepfakeStudent(pretrained=False,embed_size=384,frames=frames)
 model.load_state_dict(torch.load(weights,map_location='cpu'))
 model.eval()
 example=torch.randn(1,frames,3,224,224)
 traced=torch.jit.trace(model,example)
 if apply_dynamic_quant:
  try:
   qmodel=torch.quantization.quantize_dynamic(traced,{torch.nn.Linear},dtype=torch.qint8)
   traced=qmodel
  except Exception as e:
   print('dynamic quant failed',e)
 if apply_mobile_opt:
  try:
   traced=optimize_for_mobile(traced)
  except Exception as e:
   print('mobile optimize failed',e)
 traced.save(out)
 print('Saved',out)
if __name__=='__main__':
 export_torchscript()
