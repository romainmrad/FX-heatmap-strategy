from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.data.heatmap_builder import HeatmapBuilder


if __name__ == '__main__':
    loader = DataLoader('./config/universe.ini')
    loader.download_data()
    processor = DataProcessor('./config/processor.ini')
    processor.process()
    builder = HeatmapBuilder('./config/builder.ini')
    builder.build()
