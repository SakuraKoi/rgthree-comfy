import folder_paths

from typing import Union

import comfy
import comfy.lora
import comfy.utils
import folder_paths

from nodes import LoraLoader
from .constants import get_category, get_name
from .power_prompt_utils import get_lora_by_filename
from .utils import FlexibleOptionalInputType, any_type
from .server.utils_info import get_model_info_file_data
from .log import log_node_warn

NODE_NAME = get_name('Power Lora Stacker')


class RgthreePowerLoraStacker:
  """ The Power Lora Stacker is a powerful, flexible node to load multiple loras to be used with LoRA-Merger-ComfyUI."""

  NAME = NODE_NAME
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
      },
      # Since we will pass any number of loras in from the UI, this needs to always allow an
      "optional": FlexibleOptionalInputType(type=any_type, data={
        "model": ("MODEL",),
        "clip": ("CLIP",),
      }),
      "hidden": {},
    }

  RETURN_TYPES = ("LoRAStack", "LoRAWeights")
  RETURN_NAMES = ("LoRAStack", "LoRAWeights")
  FUNCTION = "load_loras"

  def load_loras(self, model=None, clip=None, **kwargs):
    """Loops over the provided loras in kwargs and applies valid ones."""

    key_map = {}
    if model is not None:
      key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
      key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    
    
    lora_patch_dicts = {}
    lora_strengths = {}

    loaded_count = 0

    for key, value in kwargs.items():
      key = key.upper()
      if key.startswith('LORA_') and 'on' in value and 'lora' in value and 'strength' in value:
        strength_model = value['strength']
        # If we just passed one strength value, then use it for both, if we passed a strengthTwo
        # as well, then our `strength` will be for the model, and `strengthTwo` for clip.
        strength_clip = value['strengthTwo'] if 'strengthTwo' in value else None
        if clip is None:
          if strength_clip is not None and strength_clip != 0:
            log_node_warn(NODE_NAME, 'Recieved clip strength eventhough no clip supplied!')
          strength_clip = 0
        else:
          strength_clip = strength_clip if strength_clip is not None else strength_model
        if value['on'] and (strength_model != 0 or strength_clip != 0):
          lora_name = value['lora']
          lora_file = get_lora_by_filename(lora_name, log_node=self.NAME)
          if lora_file is not None:
            try:
              lora_path = folder_paths.get_full_path("loras", lora_file)
              lora_raw = comfy.utils.load_torch_file(lora_path, safe_load=True)
              lora_name_base = os.path.splitext(os.path.basename(lora_file))[0]
              lora_name_pretty = lora_name_base
              collision_count = 1
              while lora_name_pretty in lora_patch_dicts:
                print(f"[{self.NAME}] WARNING: LoRA name collision detected for '{lora_name_base}'. Using '{lora_name_base}_{collision_count}' instead.")
                lora_name_pretty = f"{lora_name_base}_{collision_count}"
                collision_count += 1
              patch_dict = comfy.lora.load_lora(lora_raw, key_map)
              lora_patch_dicts[lora_name_pretty] = patch_dict
              lora_strengths[lora_name_pretty] = {
                'strength_model': strength_model,
                'strength_clip': strength_clip,
              }
              loaded_count += 1
              print(f"[{self.NAME}] Loaded LoRA '{lora_name_pretty}' with strength {strength_model} (model) and {strength_clip} (clip) ({len(patch_dict)} layers)")
            except Exception as e:
              print(f"[{self.NAME}] Error loading LoRA '{lora_name}': {e}")
              continue

    print(f"[{self.NAME}] Loaded {loaded_count} LoRAs: {list(lora_patch_dicts.keys())}")
    return (lora_patch_dicts, lora_strengths)

  @classmethod
  def get_enabled_loras_from_prompt_node(cls,
                                         prompt_node: dict) -> list[dict[str, Union[str, float]]]:
    """Gets enabled loras of a node within a server prompt."""
    result = []
    for name, lora in prompt_node['inputs'].items():
      if name.startswith('lora_') and lora['on']:
        lora_file = get_lora_by_filename(lora['lora'], log_node=cls.NAME)
        if lora_file is not None:  # Add the same safety check
          lora_dict = {
            'name': lora['lora'],
            'strength': lora['strength'],
            'path': folder_paths.get_full_path("loras", lora_file)
          }
          if 'strengthTwo' in lora:
            lora_dict['strength_clip'] = lora['strengthTwo']
          result.append(lora_dict)
    return result

  @classmethod
  def get_enabled_triggers_from_prompt_node(cls, prompt_node: dict, max_each: int = 1):
    """Gets trigger words up to the max for enabled loras of a node within a server prompt."""
    loras = [l['name'] for l in cls.get_enabled_loras_from_prompt_node(prompt_node)]
    trained_words = []
    for lora in loras:
      info = get_model_info_file_data(lora, 'loras', default={})
      if not info or not info.keys():
        log_node_warn(
          NODE_NAME,
          f'No info found for lora {lora} when grabbing triggers. Have you generated an info file'
          ' from the Power Lora Loader "Show Info" dialog?'
        )
        continue
      if 'trainedWords' not in info or not info['trainedWords']:
        log_node_warn(
          NODE_NAME,
          f'No trained words for lora {lora} when grabbing triggers. Have you fetched data from'
          'civitai or manually added words?'
        )
        continue
      trained_words += [w for wi in info['trainedWords'][:max_each] if (wi and (w := wi['word']))]
    return trained_words
