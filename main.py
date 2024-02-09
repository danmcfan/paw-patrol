from PIL import ImageDraw

from src.data import create_images

from src.model import train, predict

version = 4
filename = f"model_v{version}"

train(filename, batch_size=50, num_epochs=10, count=1000)

images = create_images()

for index, image in enumerate(images[:10]):
    pil_image = image.get_pil_image()

    draw = ImageDraw.Draw(pil_image)

    for obj in image.objects:
        draw.rectangle(
            [
                (obj.bounding_box.xmin, obj.bounding_box.ymin),
                (obj.bounding_box.xmax, obj.bounding_box.ymax),
            ],
            outline="green",
            width=2,
        )

    for box, label, score in predict(filename, image, threshold=0.9):
        draw.rectangle(
            [
                (box[0], box[1]),
                (box[2], box[3]),
            ],
            outline="red",
            width=2,
        )
        draw.text(
            (box[0], box[1]),
            f"{score:.2f}",
            fill="black",
        )

    pil_image.save(f"data/outputs/{filename}/{index}.jpg")
