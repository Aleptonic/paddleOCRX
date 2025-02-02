
import os
from PDFSNIPPER import save_pages_as_images
# Set the path to the poppler binaries
os.environ['PATH'] += os.pathsep + r'C:\Program Files\Release-24.08.0-0\poppler-24.08.0\Library\bin'
save_pages_as_images('input','output_reports',[0])
