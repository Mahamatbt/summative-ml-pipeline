import os
import random
from locust import HttpUser, task, between

# We need a sample image to send to the /predict endpoint.
# Since locust runs locally, we can just create a dummy one if it doesn't exist.
DUMMY_IMAGE_PATH = "dummy_image.jpg"

if not os.path.exists(DUMMY_IMAGE_PATH):
    from PIL import Image
    # Create a random RGB image
    img = Image.new('RGB', (160, 160), color = (73, 109, 137))
    img.save(DUMMY_IMAGE_PATH)

class GarbageClassificationUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def check_uptime(self):
        self.client.get("/uptime")

    @task(5)
    def predict_image(self):
        with open(DUMMY_IMAGE_PATH, "rb") as image:
            self.client.post(
                "/predict",
                files={"file": ("dummy.jpg", image, "image/jpeg")}
            )
