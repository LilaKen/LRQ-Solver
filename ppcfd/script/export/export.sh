# python main.py --config-path ./ppcfd/script/export/ --config-name gino.yaml
# python main.py --config-path ./ppcfd/script/export/ --config-name transolver.yaml
# python main.py --config-path ./ppcfd/script/export/ --config-name transolver_vel.yaml

# python infer.py --config-path ./ppcfd/script/export/ --config-name gino.yaml
python infer.py --config-path ./ppcfd/script/export/ --config-name transolver.yaml
# python infer.py --config-path ./ppcfd/script/export/ --config-name transolver_vel.yaml
