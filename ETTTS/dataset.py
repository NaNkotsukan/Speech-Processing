import numpy as np

def batch_make(batchsize,max_num=2400):
    texts=[]
    mels=[]
    inputs=[]
    mellens=[]
    textlens=[]
    for _ in range(batchsize):
        mel_name='BASIC5000_'+str(number)+'_mel.npy'
        txt_name='BASIC5000_'+str(number)+'_txt.npy'

        mel=np.load(mel_name)
        mel=mel.astype(np.float32)
        mels.append(mel)
        mellens.append(mel.shape[1])

        x=np.zeros((mel.shape[0],1)).astype(np.float32)
        x=np.concatenate((x,mel[:,:-1]),axis=1)
        inputs.append(x)

        text=np.load(txt_name)
        text=text.astype(np.int32)
        texts.append(text)
        textlens.append(text.shape[0])

        cmels-[]
        ctxts=[]
        cxs=[]

    for mel,x,txt in zip(mels, inputs, texts):
        cmel=np.pad(mel,(0,max(mellens)-mel.shape[1]),'constant',constant_values=0)
        cx=np.pad(x,max(mellens)-cx.shape[0],'constant',constant_values=0)
        ctxt=np.pad(txt,(0,max(textlens)-txt.shape[0]),'constant',constant_values=31)

        cmels.append(cmel)
        ctxts.append(ctxt)
        cxs.append(cx)

    mels=np.array(cmels)
    inputs=np.array(cxs)
    txts=np.array(ctxts)

    return txts,inputs,mels,textlens,mellens
