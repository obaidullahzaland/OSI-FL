# Incremental One Shot Federated Learning


# OSI-FL
This repository hosts the supplementary code for our paper titled "One Shot Incremental Federated Learning with Selective Sample Retention".

# Requirements

    pip install torch torchvision
    
# Train

    # FL
    python train_federated.py \
        --methods fedavg \
        --dataset "nico_dg" \
        --backbone "resnet18" \

    # OSI-FL
    python train.py \
        --model topk \
        --dataset "nico_dg" \
        --backbone "resnet18" \
        --k 5 \



# Acknowledgements
This work is based on 

(1) [OSCAR](https://github.com/obaidullahzaland/oscar)


