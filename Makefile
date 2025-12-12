shape_exp1:
	python train.py \
		--dataset=Shapenet \
		--coverageLoss=0 \
		--countLoss=0 \
		--name=shape_exp1

shape_exp2:
	python train.py \
		--dataset=Shapenet \
		--countLoss=0 \
		--name=shape_exp2

shape_exp3:
	python train.py \
		--dataset=Shapenet \
		--name=shape_exp3

shape_exp4:
	python train.py \
		--dataset=Shapenet \
		--countLoss=0 \
		--name=shape_exp4 \
		--nParts=2

infer_shape_exp1:
	python inference.py \
		--dataset=Shapenet \
		--checkpoint=output/snapshots/shape_exp1/iter9.pkl \
		--name=shape_exp1

infer_shape_exp2:
	python inference.py \
		--dataset=Shapenet \
		--checkpoint=output/snapshots/shape_exp2/iter9.pkl \
		--name=shape_exp2

infer_shape_exp3:
	python inference.py \
		--dataset=Shapenet \
		--checkpoint=output/snapshots/shape_exp3/iter9.pkl \
		--name=shape_exp3

infer_shape_exp4:
	python inference.py \
		--dataset=Shapenet \
		--checkpoint=output/snapshots/shape_exp4/iter9.pkl \
		--name=shape_exp4