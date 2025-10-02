from src.utils.dir_utils import init_project, remove_ds_store
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
# from src.data.heatmap_builder import HeatmapBuilder
# from src.strategy.thermal_vision import ThermalVision
# from src.strategy.backtest import BackTest


if __name__ == '__main__':
    init_project()
    remove_ds_store()
    loader = DataLoader('config/data_loader.ini')
    loader.load()
    processor = DataProcessor('config/data_processor.ini')
    processor.process()
    # builder = HeatmapBuilder('config/heatmap_builder.ini')
    # builder.build()
    # vision = ThermalVision('config/thermal_vision.ini')
    # vision.train()
    # backtest = BackTest('config/backtest.ini', vision)
    # backtest.run()
