from facenet_pytorch import MTCNN
import os,cv2,torch,random
mt=MTCNN(image_size=224,keep_all=False,device='cuda' if torch.cuda.is_available() else 'cpu')
def ensure(d): 
 if not os.path.exists(d): os.makedirs(d)
def extract_faces_from_video(video_path,out_dir,frames_per_video=8,max_per_video=None):
 cap=cv2.VideoCapture(video_path)
 total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 if total<=0: return 0
 step=max(1,total//frames_per_video)
 saved=0
 for i in range(0,total,step):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i)
  ret,frame=cap.read()
  if not ret: break
  img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  try:
   face=mt(img)
  except Exception:
   face=None
  if face is None: continue
  face=face.permute(1,2,0).numpy()
  out_path=os.path.join(out_dir,f"{os.path.splitext(os.path.basename(video_path))[0]}_{i}.jpg")
  face=cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
  cv2.imwrite(out_path,face)
  saved+=1
  if max_per_video and saved>=max_per_video: break
 return saved

def build_crops(src_root='data/raw',out_root='data/crops',frames_per_video=8,train_ratio=0.7,val_ratio=0.15):
 ensure(out_root)
 classes=['real','fake']
 all_files=[]
 for cls in classes:
  src_dir=os.path.join(src_root,cls)
  out_dir=os.path.join(out_root,cls)
  ensure(out_dir)
  if not os.path.exists(src_dir): continue
  for f in os.listdir(src_dir):
   full=os.path.join(src_dir,f)
   if not os.path.isfile(full): continue
   extract_faces_from_video(full,out_dir,frames_per_video)
  for fn in os.listdir(out_dir):
   all_files.append((os.path.join(out_dir,fn),1 if cls=='fake' else 0))
 random.shuffle(all_files)
 n=len(all_files)
 ntrain=int(n*train_ratio)
 nval=int(n*val_ratio)
 train=all_files[:ntrain]
 val=all_files[ntrain:ntrain+nval]
 test=all_files[ntrain+nval:]
 def write_split(lst,fn):
  with open(fn,'w') as fh:
   for p,lbl in lst:
    fh.write(f"{p},{lbl}\n")
 write_split(train,'data/crops/train.txt')
 write_split(val,'data/crops/val.txt')
 write_split(test,'data/crops/test.txt')
 return {'total_images':n,'train':len(train),'val':len(val),'test':len(test)}
if __name__=='__main__':
 print(build_crops())
