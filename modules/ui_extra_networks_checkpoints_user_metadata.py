import gradio as gr

from modules import ui_extra_networks_user_metadata, sd_samplers, sd_vae, shared, shared_items
from modules.ui_components import ToolButton
from modules_forge import main_entry

refresh_symbol = '\U0001f504'  # 🔄

class CheckpointUserMetadataEditor(ui_extra_networks_user_metadata.UserMetadataEditor):
    def __init__(self, ui, tabname, page):
        super().__init__(ui, tabname, page)

        self.select_vae = None
        self.sd_version = 'Unknown'
        self.gen_steps = None
        self.gen_sampler = None
        self.gen_scheduler = None
        self.gen_cfg = None
        self.gen_distilled_cfg = None
        self.gen_width = None
        self.gen_height = None

    def save_user_metadata(self, name, desc, notes, vae, sd_version,
                           gen_steps, gen_sampler, gen_scheduler,
                           gen_cfg, gen_distilled_cfg, gen_width, gen_height):
        user_metadata = self.get_user_metadata(name)
        user_metadata["description"] = desc
        user_metadata["notes"] = notes
        user_metadata["vae_te"] = vae
        user_metadata["sd_version_str"] = 'SdVersion.' + sd_version

        gen_params = {}
        if gen_steps and gen_steps > 0:
            gen_params["steps"] = int(gen_steps)
        if gen_sampler:
            gen_params["sampler"] = gen_sampler
        if gen_scheduler:
            gen_params["scheduler"] = gen_scheduler
        if gen_cfg and gen_cfg > 0:
            gen_params["cfg"] = float(gen_cfg)
        if gen_distilled_cfg and gen_distilled_cfg > 0:
            gen_params["distilled_cfg"] = float(gen_distilled_cfg)
        if gen_width and gen_width > 0:
            gen_params["width"] = int(gen_width)
        if gen_height and gen_height > 0:
            gen_params["height"] = int(gen_height)

        if gen_params:
            user_metadata["gen_params"] = gen_params
        else:
            user_metadata.pop("gen_params", None)

        self.write_user_metadata(name, user_metadata)

    def put_values_into_components(self, name):
        user_metadata = self.get_user_metadata(name)
        values = super().put_values_into_components(name)

        vae = user_metadata.get('vae_te', None)
        if vae is None:     # fallback to old type
            vae = user_metadata.get('vae', None)
            if vae is not None:
                if isinstance(vae, str):
                    vae = [vae]

        version = user_metadata.get('sd_version_str', '')
        if version == '':
            version = 'Unknown'
        else:
            version = version.replace('SdVersion.', '')

        gen_params = user_metadata.get('gen_params', {})

        return [
            *values[0:5],
            vae,
            version,
            gen_params.get("steps", 0),
            gen_params.get("sampler", ""),
            gen_params.get("scheduler", ""),
            gen_params.get("cfg", 0.0),
            gen_params.get("distilled_cfg", 0.0),
            gen_params.get("width", 0),
            gen_params.get("height", 0),
        ]

    def create_editor(self):    #happens before main_entry.modules_list is filled
        modules_list = ['Built in']
        if main_entry.module_list == {}:
            _, modules = main_entry.refresh_models()
            modules_list += list(modules)
        else:
            modules_list += list(main_entry.module_list.keys())

        def refreshModules ():
            return gr.update(choices=['Built in'] + list(main_entry.module_list.keys()))

        self.create_default_editor_elems()

        self.sd_version = gr.Radio(['SD1', 'SDXL', 'Flux', 'Unknown'], value='Unknown', label='Base model', interactive=True)

        with gr.Row():
            self.select_vae = gr.Dropdown(choices=modules_list, value=None, label="Preferred VAE / Text encoder(s)", elem_id="checpoint_edit_user_metadata_preferred_vae", multiselect=True)
            self.refresh = ToolButton(refresh_symbol)

            self.refresh.click(fn=refreshModules, outputs=self.select_vae, show_progress='hidden')

        self.edit_notes = gr.TextArea(label='Notes', lines=4)

        sampler_choices = [""] + [x.name for x in sd_samplers.all_samplers]
        scheduler_choices = [""] + shared_items.list_schedulers()

        with gr.Accordion("Generation Parameters", open=False):
            gr.Markdown("Override generation parameters when this checkpoint is loaded. Leave at 0 / blank to use preset defaults.")
            with gr.Row():
                self.gen_steps = gr.Number(label="Steps", value=0, minimum=0, maximum=150, precision=0)
                self.gen_cfg = gr.Number(label="CFG Scale", value=0.0, minimum=0.0, maximum=30.0, step=0.5)
                self.gen_distilled_cfg = gr.Number(label="Distilled CFG", value=0.0, minimum=0.0, maximum=10.0, step=0.5)
            with gr.Row():
                self.gen_sampler = gr.Dropdown(label="Sampler", choices=sampler_choices, value="")
                self.gen_scheduler = gr.Dropdown(label="Scheduler", choices=scheduler_choices, value="")
            with gr.Row():
                self.gen_width = gr.Number(label="Width", value=0, minimum=0, maximum=4096, precision=0)
                self.gen_height = gr.Number(label="Height", value=0, minimum=0, maximum=4096, precision=0)

        self.create_default_buttons()

        viewed_components = [
            self.edit_name,
            self.edit_description,
            self.html_filedata,
            self.html_preview,
            self.edit_notes,
            self.select_vae,
            self.sd_version,
            self.gen_steps,
            self.gen_sampler,
            self.gen_scheduler,
            self.gen_cfg,
            self.gen_distilled_cfg,
            self.gen_width,
            self.gen_height,
        ]

        self.button_edit\
            .click(fn=self.put_values_into_components, inputs=[self.edit_name_input], outputs=viewed_components)\
            .then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[self.box])

        edited_components = [
            self.edit_description,
            self.edit_notes,
            self.select_vae,
            self.sd_version,
            self.gen_steps,
            self.gen_sampler,
            self.gen_scheduler,
            self.gen_cfg,
            self.gen_distilled_cfg,
            self.gen_width,
            self.gen_height,
        ]

        self.setup_save_handler(self.button_save, self.save_user_metadata, edited_components)
