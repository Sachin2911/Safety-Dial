# LeWM Experiment1 compute template

`vast-template.json` is the source for a private Vast launch template.

We use: `vastai/pytorch:2.7.1-cuda-12.6.3-py311-22.04` image.

## Settings

| Setting | Initial value |
| --- | --- |
| Access | Direct SSH, using the account's registered SSH public key |
| GPU selection | One RTX 4090 on a verified amd64 host |
| Driver | At least 560.35.05 and advertised CUDA support of at least 12.6 |
| CPU | At least 8 effective vCPUs |
| RAM search filter | At least 64GB |
| Disk | 200 GB |
| Network search floor | 1,000 Mbps download |
| Historical reliability filter | At least 0.98, not an uptime guarantee |
| Startup | `/opt/instance-tools/bin/entrypoint.sh` |
| Python | Use `/venv/main/bin/python` explicitly |
| Rendering settings | OSMesa/llvmpipe and dummy SDL, matching the AP reference |
| Hugging Face cache | `/workspace/LeWM_Experiment1/cache/huggingface` |

The rendering environment variables do not install OSMesa or AP. Those libraries
will be baked into the custom image. The script does not install packages, fetch
models or start research jobs. Rental shutdown remains manual.
