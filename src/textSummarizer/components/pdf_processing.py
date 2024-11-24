from src.textSummarizer.logging import logger
from src.textSummarizer.components.pdf_extraction import iter_text_per_page, categorize_pdf, process_large_pdf
from src.textSummarizer.utils.retry import retry
from src.textSummarizer.entity import PdfProcessingConfig
from pathlib import Path
from dask.diagnostics import ProgressBar
from dask import delayed, compute

class PdfProcessing:
    def __init__(self, config: PdfProcessingConfig):
        self.config = config

    def process_pdf(self, pdf_path: Path):
        @retry(max_retries=self.config.max_retries, wait_time=self.config.wait_time)
        def _process():
            """Processes a single PDF, extracting its text."""
            try:
                category, page_count = categorize_pdf(pdf_path=pdf_path)
                if category == 'Error':
                    logger.error(f"Failed to process {pdf_path}: PDF could not be categorized.")
                    return None  # Skip corrupted PDFs

                # Extract text based on category
                if category == 'Long':
                    text = process_large_pdf(pdf_path=pdf_path, page_count=page_count, batch_size=self.config.batch_size)
                else:
                    text = "\n".join([page_text for _, page_text in iter_text_per_page(pdf_path)])

                return category, page_count, text  # Return the extracted text

            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                return None  # Return None to handle failed PDF processing gracefully
        return _process()

    def process_multiple_pdfs(self):
        """Process multiple PDFs concurrently using Dask with a progress bar and delayed tasks."""
        
        pdf_folder_path = self.config.pdf_folder_path
        pdf_paths = list(Path(pdf_folder_path).glob("*.pdf"))
        
        # Create delayed tasks for each PDF processing
        tasks = [delayed(self.process_pdf)(pdf_path) for pdf_path in pdf_paths]

        # Progress bar using dask's diagnostics
        with ProgressBar():
            results = compute(*tasks, scheduler='processes')  # Using the 'threads' scheduler for concurrency

        # Iterate through the results and handle errors
        for result in results:
            if "error" in result:
                logger.error(f"Error processing PDF: {result['error']}")
            else:
                logger.info(f"PDF processed successfully. Path: {len(result[2])}")
        import pandas as pd
        pd.DataFrame(results,columns=['category','page_count','text']).to_csv('temptext.csv',index=False)
        df = pd.read_csv('metadata.csv')
        ordered_results_idx = [(idx, df.index[df['file_name']==path.name][0]) for idx, path in enumerate(pdf_paths)]
        ordered_results = [0]*len(df)
        for i in ordered_results_idx:
            ordered_results[i[1]] = results[i[0]]
        ordered_results_upload = []
        ordered_results_return = []
        for tuple in ordered_results:
            ordered_results_upload.append(tuple[:2])
            ordered_results_return.append(tuple[-1])
        df = pd.concat([df,pd.DataFrame(ordered_results_upload,columns=['category','page_count'])],axis=1)
        df.to_csv('text.csv', index=False)
        return ordered_results_return
    
    
        
    


