import numpy as np
from torch import dtype


max_frame = 300
num_joint = 25
num_person_out = 2

with open('/media/mtk/559be1a1-ed84-4771-ac8e-97e50374afcd/Workspace/pkummd/labels/0002-L.txt', 'r') as f:
    all_labels = f.readline()

all_labels = all_labels.split(',')
print(all_labels)

start_frame = int(all_labels[1])
end_frame = int(all_labels[2])
print(start_frame, end_frame)

with open('/media/mtk/559be1a1-ed84-4771-ac8e-97e50374afcd/Workspace/pkummd/skeletons/0002-L.txt', 'r') as f:
    all_skeletons = f.readlines()

fp = np.zeros((3, max_frame, num_joint, num_person_out), dtype=np.float32)

data = []
frames_data = []
l = 0
for i, two_ppl_skeletons in enumerate(all_skeletons):
    #print(i)
    if i >= start_frame and i < end_frame:
        two_ppl_skeletons = two_ppl_skeletons.split()
        #print(two_ppl_skeletons)#, two_ppl_skeletons.shape)
        fst_person = np.array(two_ppl_skeletons[:75], dtype=np.float16)
        scd_person = np.array(two_ppl_skeletons[75:], dtype=np.float16)
        data = []
        for m, person in enumerate([fst_person, scd_person]):
            #for kpoints in range(25):
            #print('person ',person)
            x = person[::3]
            y = person[1::3]
            z = person[2::3]
            for n, points in enumerate([x,y,z]):
                for o, point in enumerate(points):
                    #print(point)
                    fp[n, l, o, m] = point
        l += 1
            #person_kpoints = np.stack([x,y,z])
            #print('person kpoints ',person_kpoints.shape, person_kpoints)
            #data.append(person_kpoints)
        #print(data)
        #data = np.array(data)
        #frames_data.append(data)
        #print(data.shape, data)
    elif i == end_frame:
        print(fp.shape)
        print()
        start_frame = 
        end_frame = 
        # frames_data = np.array(frames_data)
        # print('frames data',frames_data.shape, frames_data[0])
        # frames_data = frames_data.transpose(2,0,3,1)
        # print(frames_data.shape, frames_data[0])
        # print(fst_person.shape, scd_person.shape)
        # print(fst_person)
        # print(scd_person)
        # two_ppl_skeletons = np.stack([fst_person, scd_person])
        # print('two ppl skeletons ',two_ppl_skeletons.shape)
        #two_ppl_skeletons.len
        # for person in len(two_ppl_skeletons):
        #     print(len(fst_person), len(scd_person))
    #elif i == end_frame:
    #    print(i)