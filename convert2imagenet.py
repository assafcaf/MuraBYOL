import tensorflow as tf
import tensorflow_datasets as tfds

pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MURATfds"
builder = tfds.ImageFolder(pth)
print(builder.info)  # num examples, labels... are automatically calculated
ds = builder.as_dataset(split='train', shuffle_files=True)
tfds.show_examples(ds, builder.info)

