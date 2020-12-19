import os
import geopandas as gpd
import matplotlib.pyplot as plt

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
urkaine = world[world["continent"] == "Ukraine"]
capitals = gpd.read_file(gpd.datasets.get_path("naturalearth_cities"))

config = {'data_dir': '/home/val/Downloads/noise_urk',
		'city': 'lviv'}

def list_files():
	data_dir = config['data_dir']
	files = [os.path.join(
		data_dir, x) for x in os.listdir(data_dir)]

	return files

def read_gjson():
	files = list_files()
	df = gpd.read_file(files[0])
	print(df.head())
	return df

def plot_data(poly_gdf):
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
	world.plot(ax=ax1)
	poly_gdf.boundary.plot(ax=ax1, color="red")
	urkaine.boundary.plot(ax=ax2, color="green")
	capitals.plot(ax=ax2, color="purple")
	ax1.set_axis_off()
	ax2.set_axis_off()
	plt.show()



if __name__ == '__main__':
	df = read_gjson()
	plot_data(df)