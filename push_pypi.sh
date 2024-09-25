rm -rf light_embed.egg-info
rm -rf dist
python3 -m build
twine upload dist/*