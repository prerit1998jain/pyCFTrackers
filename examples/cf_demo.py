import os
from examples.pytracker import PyTracker
from lib.utils import get_ground_truthes,plot_precision,plot_success
from examples.otbdataset_config import OTBDatasetConfig

if __name__ == '__main__':
    # Data to be installed in one folder up test folder.
    data_dir='../dataset/test'

    # Making a sorted list of the data files in the particular path
    data_names=sorted(os.listdir(data_dir))
    print(data_names)

    # An instance of the class defined in otbdataset.config
    dataset_config=OTBDatasetConfig()

    # Traversing over the data_names
    for data_name in data_names:

        # Identifying the path to a particular file.
        data_path = os.path.join(data_dir,data_name)

        # Getting the ground_truth in the form of an np array
        gts = get_ground_truthes(data_path)

        #  Why is it?
        if data_name in dataset_config.frames.keys():
            start_frame,end_frame=dataset_config.frames[data_name][:2]
            if data_name!='David':
                gts=gts[start_frame-1:end_frame]

        # Identifying the directory consisting of the images.
        img_dir = os.path.join(data_path,'img')

        # Tracker is defined (STRCF is used here.)
        tracker = PyTracker(img_dir,tracker_type='STRCF',dataset_config=dataset_config)

        # tracking method returns the coordinates of the bounding boxes in the furthur frames
        poses=tracker.tracking(verbose=True,video_path=os.path.join('../results/CF',data_name+'_vis.avi'))

        plot_success(gts,poses,os.path.join('../results/CF',data_name+'_success.jpg'))
        plot_precision(gts,poses,os.path.join('../results/CF',data_name+'_precision.jpg'))
