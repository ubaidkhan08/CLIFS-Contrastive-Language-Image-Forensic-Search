import os
import clip
import torch
import cv2
import math
import glob
import logging
import time
from PIL import Image
import numpy as np
from patchify import patchify

class CLIFS:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO)


        # Choose device and load the chosen model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(os.getenv('MODEL'), self.device, jit=False)

        self.image_features = None
        self.feature_idx_to_video = []

        # Preload the videos in the data input directory
        # This is done as upload through web interface isn't implemented yet
        for file in glob.glob('{}/*'.format(os.getenv('INPUT_DIR'))):
            self.add_video(file)


    def add_video(self, path, batch_size=512, ms_between_features=1000,
                  patch_size=360):
        # Calculates features from video images.
        # Loops over the input video to extract every frames_between_features
        # frame and calculate the features from it. The features are saved
        # along with mapping of what video and frame each detection
        # corresponds to.
        # The actual batch size can be up to batch_size + number of patches.
        logging.info('Adding video: {}'.format(path))
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_between_features = fps / (1000 / ms_between_features)
        feature_list = []
        feature_video_map = []

        frame_idx = 0
        to_encode = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frames_between_features == 0:
                patches = self._make_patches(frame, patch_size) + [frame]
                for idx, patch in enumerate(patches):
                    feature_data = {'video_path': path,
                                    'frame_idx': frame_idx,
                                    'time': frame_idx / fps}
                    feature_video_map.append(feature_data)
                    to_encode.append(patch)
            if len(to_encode) >= batch_size:
                image_features = self._calculate_images_features(to_encode)
                feature_list.append(image_features)
                to_encode = []

            frame_idx += 1
        if len(to_encode) > 0:
            image_features = self._calculate_images_features(to_encode)
            feature_list.append(image_features)
        feature_tensor = torch.cat(feature_list, dim=0)
        self._add_image_features(feature_tensor, feature_video_map)


    def _make_patches(self, frame, patch_size):
        # To get more information out of images, we divide the image
        # into smaller patches that are closer to the input size of the model
        step = int(patch_size / 2)
        patches_np = patchify(frame, (patch_size, patch_size, 3),
                              step=step)
        patches = []
        for i in range(patches_np.shape[0]):
            for j in range(patches_np.shape[1]):
                patches.append(patches_np[i, j, 0])
        return patches



    def _calculate_images_features(self, images):
        # Preprocess an image, send it to the computation device and perform
        # inference
        logging.info(f'Calculating features for batch of {len(images)} frames')
        for i in range(len(images)):
            start_time = time.time()
            images[i] = self._preprocess_image(images[i])
            end_time = time.time()
            logging.debug(f"Preprocessing image took {end_time-start_time} seconds")
        image_stack = torch.stack(images, dim=0)
        image_tensor = image_stack.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
        return image_features


    def _preprocess_image(self, image):
        # cv2 image to PIL image to the model's preprocess function
        # which makes sure the image is ok to ingest and makes it a tensor
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_tensor = self.preprocess(image_pil)
        return image_tensor


    def _add_image_features(self, new_features, feature_video_map):
        # Controls the addition of image features to the object
        # such that video mappings are provided, etc.,
        assert(new_features.shape[0] == len(feature_video_map))
        new_features /= new_features.norm(dim=-1, keepdim=True)
        if self.image_features is not None:
            self.image_features = torch.cat((self.image_features, new_features),
                                            dim=0)
        else:
            self.image_features = new_features
        self.feature_idx_to_video.extend(feature_video_map)


	def search(self, query, n=9, threshold=37):
	    """
	    Finds the most similar images corresponding to a given query.

	    Args:
	        query (str): The text query to search for.
	        n (int, optional): The maximum number of results to return. Defaults to 9.
	        threshold (int, optional): The minimum score threshold to consider a match. Defaults to 37.

	    Returns:
	        A list of dicts containing metadata about the matching frames, including the video path,
	        frame index, score, and image path.
	    """
	    text_inputs = torch.cat([self.clip.tokenize(query)]).to(self.device)
	    with torch.no_grad():
	        text_features = self.model.encode_text(text_inputs)
	    text_features /= text_features.norm(dim=-1, keepdim=True)
	    similarity = (100.0 * text_features @ self.image_features.T)

	    values, indices = similarity[0].topk(n * 100)

	    used_images = set()
	    response_matches = []
	    for indices_idx, similarity_idx in enumerate(indices):
	        if len(response_matches) >= n:
	            break
	        initial_match_data = self.feature_idx_to_video[similarity_idx]
	        score = float(values[indices_idx].cpu().numpy())
	        img_hash = f"{initial_match_data['video_path']}-{initial_match_data['frame_idx']}"
	        if img_hash in used_images:
	            continue

	        if score < threshold:
	            if len(response_matches) == 0:
	                logging.info("No matches with score >= threshold found")
	            break

	        image_path = self._write_image_from_match(initial_match_data)
	        if image_path is None:
	            continue
	        full_match_data = {
	            **initial_match_data,
	            "score": score,
	            "image_path": image_path,
	        }
	        logging.info(f"Frame ({query}): {full_match_data}")
	        response_matches.append(full_match_data)
	        used_images.add(img_hash)
	    return response_matches


	def _write_image_from_match(self, match):
	    """
	    Writes an image corresponding to a given match to disk.

	    Args:
	        match (dict): A dict containing metadata about the match, including the video path,
	        frame index, and time.

	    Returns:
	        The name of the written image file, or None if the image could not be written.
	    """
	    path, ext = os.path.splitext(match["video_path"])
	    video_name = os.path.splitext(os.path.basename(path))[0]
	    image_name = f"{video_name}-{match['frame_idx']}.jpg"
	    image_path = f"{os.getenv('OUTPUT_DIR')}/{image_name}"
	    if os.path.exists(image_path):
	        return image_name
	    cap = cv2.VideoCapture(match["video_path"])
	    cap.set(cv2.CAP_PROP_POS_MSEC, match["time"] * 1000)
	    ret, frame = cap.read()
	    if not ret:
	        return None
	    cv2.imwrite(image_path, frame)
	    return image_name
