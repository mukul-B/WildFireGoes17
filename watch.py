import time
from sklearn import set_config
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from AutoEncoderEvaluation import RuntimeDLTransformation
from RealTimeTransformation import plot_prediction

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            # Trigger your program here
            # For example, call a function
            # process_new_file(event.src_path)
            # plot_prediction(dir + gfile,pathC,epsg,plotPredition,supr_resolution)

def monitor_folder(path_to_watch):
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    folder_to_monitor = "DataRepository/RealTimeIncoming_files/DavisCreekFire"
    plotPredition = True
    supr_resolution = RuntimeDLTransformation(set_config) if plotPredition == True else None
    monitor_folder(folder_to_monitor)
