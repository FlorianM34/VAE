import tensorflow as tf

# Vérifier les GPUs physiques disponibles
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("Aucun GPU détecté. Assurez-vous que TensorFlow est installé avec le support GPU et que les pilotes NVIDIA sont correctement configurés.")
else:
    print("Nombre de GPUs physiques disponibles:", len(physical_devices))
    for gpu in physical_devices:
        print("Nom du GPU:", gpu.name)
        print("Type du GPU:", gpu.device_type)

# Vérifier les périphériques logiques (comment TensorFlow utilise les GPUs)
logical_devices = tf.config.experimental.list_logical_devices('GPU')
for device in logical_devices:
    print(device)

