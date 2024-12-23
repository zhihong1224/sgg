import os
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['font.sans-serif'] = ['SimHei']

# relations = ['on','has','wearing','of','in','near','with','behind','holding','above',
#              'sitting on','wears','under','riding','in front of','standing on',
#              'at','attached to','carrying','walking on','over','belonging to',
#              'for','looking at','watching','hanging from','parked on','laying on',
#              'eating','and','using','covering','between','covered in','along','part of',
#              'lying on','on back of','to','mounted on','walking in','across','against',
#              'from','growing on','painted on','made of','playing','says','flying in']

# rsfa_res = {'above':0.1381,'across':0.3823,'against':0.1855,'along':0.3891,'and':0.3028,'at':0.6228,
#             'attached to':0.0794,'behind':0.5795,'belonging to':0.2686,'between':0.1646,'carrying':0.7147,
#             'covered in':0.6246,'covering':0.5591,'eating':0.7749,'flying in':0.0000,'for':0.3227,
#             'from':0.2873,'growing on':0.4216,'hanging from':0.4535,'has':0.7587,'holding':0.4813,
#             'in':0.3183,'in front of':0.2762,'laying on':0.3360,'looking at':0.2540,'lying on':0.1531,
#             'made of':0.3750,'mounted on':0.2670,'near':0.1343,'of':0.5314,'on':0.3964,'on back of':0.3251,
#             'over':0.3423,'painted on':0.2802,'parked on':0.9003,'part of':0.0314, 'playing':0.0000,
#             'riding':0.8856,'says':0.0833,'sitting on':0.4077,'standing on':0.2695,'to':0.5205,
#             'under':0.4588,'using':0.3742,'walking in':0.4718,'walking on':0.6814,'watching':0.6757,
#             'wearing':0.8959,'wears':0.0786,'with':0.1275}

# motif_res =  {'above':0.1755,'across':0.0000,'against':0.0000,'along':0.0092,'and':0.0118,'at':0.2924,
#             'attached to':0.0009,'behind':0.5925,'belonging to':0.0000,'between':0.0069,'carrying':0.2683,
#             'covered in':0.0964,'covering':0.0263,'eating':0.2819,'flying in':0.0000,'for':0.0332,
#             'from':0.0000,'growing on':0.0000,'hanging from':0.0325,'has':0.8099,'holding':0.7178,
#             'in':0.3819,'in front of':0.1132,'laying on':0.0045,'looking at':0.1047,'lying on':0.0000,
#             'made of':0.0000,'mounted on':0.0000,'near':0.4598,'of':0.6447,'on':0.8037,'on back of':0.0000,
#             'over':0.0781,'painted on':0.0000,'parked on':0.0000,'part of':0.0000, 'playing':0.0000,
#             'riding':0.2979,'says':0.0000,'sitting on':0.2910,'standing on':0.0152,'to':0.0000,
#             'under':0.3815,'using':0.1413,'walking in':0.0000,'walking on':0.1365,'watching':0.2790,
#             'wearing':0.9693,'wears':0.0000,'with':0.1043}

# rsfas = []
# motifs = []
# for rel in relations:
#     rsfas.append(rsfa_res[rel])
#     motifs.append(motif_res[rel])


# labels = relations

# x = np.arange(len(labels))
# width = 0.4

# fig, ax = plt.subplots(figsize=(12,6),dpi=300)
# rects1 = ax.bar(x-width/2, motifs, width, label='MOTIFS',zorder=2)
# rects2 = ax.bar(x+width/2, rsfas, width, label='Proposed RSFA',zorder=2)

# ax.set_ylabel('mR@100',loc='top',rotation=0,labelpad=-30)
# ax.tick_params(length=0.1,pad=1,labelsize=24)
# ax.set_xticks(x)
# ax.set_xticklabels(labels,fontsize=24)
# ax.legend(loc=(0.4,0.995),frameon=False,ncol=2)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# # ax.tick_params(axis='both',labelsize=24)

# fig.tight_layout()
# fig.subplots_adjust(left=0.05,bottom=0.18)

# for i in [0,0.2,0.4,0.6,0.8,1.0]:
#     plt.axhline(i,c='lightgray',zorder=1)

# plt.xlim((-0.8,49.6))
# plt.xticks(rotation=90,fontsize=24)
# plt.show()
# plt.savefig('motif-rsfa.png')


# freqs= [0.01858, 0.00057, 0.00051, 0.00109, 0.00150, 0.00489, 0.00432, 0.02913, 0.00245, 0.00121, 
#                                        0.00404, 0.00110, 0.00132, 0.00172, 0.00005, 0.00242, 0.00050, 0.00048, 0.00208, 0.15608,
#                                        0.02650, 0.06091, 0.00900, 0.00183, 0.00225, 0.00090, 0.00028, 0.00077, 0.04844, 0.08645,
#                                        0.31621, 0.00088, 0.00301, 0.00042, 0.00186, 0.00100, 0.00027, 0.01012, 0.00010, 0.01286,
#                                        0.00647, 0.00084, 0.01077, 0.00132, 0.00069, 0.00376, 0.00214, 0.11424, 0.01205, 0.02958]

# freqs = sorted(freqs,reverse=True)

# print('ok')


# 绘制损失曲线
filename = '/home/wuyanhao/WorkSpace/wyh_sg/checkpoints/rsfa-precls-exmp-1e3/log.txt'
with open(filename, 'r') as f:
    data = f.readlines()
    
iters, all_losses, cvae_losses, sgg_losses = [],[],[],[]    

for item in data:
    if 'iter:' in item:
        tmps = item.split(' ')
        iter_idx = tmps.index('iter:')
        iter = int(tmps[iter_idx+1])
        iters.append(iter)
        
        loss_idx = tmps.index('loss:')
        loss = float(tmps[loss_idx+2][1:-1])
        all_losses.append(loss)
        
        cvae_idx = tmps.index('cvae_loss:')
        cvae_loss = float(tmps[cvae_idx+2][1:-1])
        cvae_losses.append(cvae_loss)
        
        sgg_idx = tmps.index('loss_rel:')
        sgg_loss = float(tmps[sgg_idx+2][1:-1])
        sgg_losses.append(sgg_loss)
        
print('len of iter:', len(iters))
print('len of all_losses:', len(all_losses))
print('len of cvae_losses:', len(cvae_losses))
print('len of sgg_losses:', len(sgg_losses))

plt.plot(iters, cvae_losses, c='g', label='loss of the CVAE')
plt.plot(iters, sgg_losses, c='b', label='loss of the Base SGG model')
plt.plot(iters, all_losses, c='r',label='loss of the RSFA')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend(loc='best')
plt.savefig('losses.png')
        