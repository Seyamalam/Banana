const fs = require('fs');
const path = require('path');
const { marked } = require('marked');
const puppeteer = require('puppeteer');

// Define the correct order of files for the research paper structure
const fileOrder = [
  'abstract.md',
  'introduction.md',
  'methodology.md',
  'training_methodology.md',
  'evaluation_methodology.md',
  'robustness.md',
  'ablation.md',
  'results_7.1_model_performance_comparison.md',
  'results_7.2_ablation_study_findings.md',
  'results_7.3_robustness_analysis.md',
  'results_7.4_deployment_metrics.md',
  'section_8_discussion.md',
  'conclusion.md'
];

// Function to read a file and return its contents
function readFile(filename) {
  try {
    return fs.readFileSync(filename, 'utf8');
  } catch (err) {
    console.warn(`Warning: Could not read ${filename}, skipping...`);
    return '';
  }
}

// Function to clean up the content
function cleanupContent(content) {
  // Remove <CURRENT_CURSOR_POSITION> tags
  content = content.replace(/<CURRENT_CURSOR_POSITION>/g, '');
  
  // Add proper spacing between sections
  content = content.replace(/(\n#{2,})/g, '\n\n$1');
  
  return content;
}

// Function to process image references in markdown
function processImageReferences(content) {
  // Extract all image references
  const imageRegex = /!\[Figure [^\]]*\]\(([^)]+)\)/g;
  const images = new Set();
  let match;
  
  while ((match = imageRegex.exec(content)) !== null) {
    const imagePath = match[1];
    if (!imagePath.startsWith('http')) {
      images.add(imagePath);
    }
  }
  
  return [...images];
}

// Function to convert images to base64
function encodeImagesToBase64(content) {
  return content.replace(/!\[Figure [^\]]*\]\(([^)]+)\)/g, (match, imagePath) => {
    if (!imagePath.startsWith('http')) {
      try {
        // Check if file exists
        const absolutePath = path.resolve(imagePath);
        if (fs.existsSync(absolutePath)) {
          // Read image file and convert to base64
          const imageBuffer = fs.readFileSync(absolutePath);
          const base64Image = imageBuffer.toString('base64');
          const extension = path.extname(imagePath).substring(1).toLowerCase();
          const mimeType = extension === 'svg' ? 'image/svg+xml' : 
                          extension === 'png' ? 'image/png' : 
                          extension === 'jpg' || extension === 'jpeg' ? 'image/jpeg' : 'image/png';
          
          // Replace path with base64 encoded data
          return match.replace(imagePath, `data:${mimeType};base64,${base64Image}`);
        } else {
          console.warn(`Warning: Image file not found: ${imagePath}`);
          return match;
        }
      } catch (err) {
        console.warn(`Warning: Error processing image ${imagePath}:`, err.message);
        return match;
      }
    }
    return match;
  });
}

// Function to generate table of contents
function generateTOC(combinedContent) {
  const lines = combinedContent.split('\n');
  let toc = '# Table of Contents\n\n';
  let sectionNum = 0;
  
  lines.forEach(line => {
    if (line.startsWith('## ')) {
      sectionNum++;
      const title = line.replace('## ', '').trim();
      const anchor = title.toLowerCase().replace(/[^\w]+/g, '-');
      toc += `${sectionNum}. [${title}](#${anchor})\n`;
    } else if (line.startsWith('### ') && !line.includes('CURRENT_CURSOR_POSITION')) {
      const title = line.replace('### ', '').trim();
      const anchor = title.toLowerCase().replace(/[^\w]+/g, '-');
      toc += `   - [${title}](#${anchor})\n`;
    }
  });
  
  return toc + '\n\n<div class="page-break"></div>\n\n' + combinedContent;
}

// Main function to combine files and generate PDF
async function generateResearchPaperPDF() {
  try {
    console.log('Reading markdown files...');
    let combinedContent = '';
    
    // Read and combine files in the specified order
    for (const filename of fileOrder) {
      console.log(`Processing ${filename}...`);
      const content = readFile(filename);
      combinedContent += content + '\n\n';
    }
    
    // Clean up the content
    combinedContent = cleanupContent(combinedContent);
    
    console.log('Processing images...');
    // Convert image paths to base64
    combinedContent = encodeImagesToBase64(combinedContent);
    
    // Generate table of contents
    combinedContent = generateTOC(combinedContent);
    
    console.log('Converting to HTML...');
    
    // Convert markdown to HTML
    const htmlContent = marked.parse(combinedContent);
    
    // Create a full HTML document with proper styling
    const fullHtml = `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Banana Leaf Disease Research Paper</title>
      <style>
        body {
          font-family: 'Arial', sans-serif;
          font-size: 12pt;
          line-height: 1.5;
          color: #333;
          margin: 2cm;
        }
        h1, h2, h3, h4 {
          color: #222;
          margin-top: 1em;
        }
        h1 { font-size: 20pt; }
        h2 { font-size: 16pt; page-break-before: always; }
        h3 { font-size: 14pt; }
        h4 { font-size: 12pt; }
        table {
          border-collapse: collapse;
          width: 100%;
          margin: 1em 0;
          page-break-inside: avoid;
        }
        th, td {
          border: 1px solid #ddd;
          padding: 8px;
        }
        th {
          background-color: #f2f2f2;
        }
        img {
          max-width: 100%;
          height: auto;
          display: block;
          margin: 1em auto;
          page-break-inside: avoid;
        }
        code {
          font-family: 'Courier New', monospace;
          background-color: #f5f5f5;
          padding: 2px 4px;
          border-radius: 3px;
        }
        pre {
          background-color: #f5f5f5;
          padding: 10px;
          border-radius: 5px;
          overflow-x: auto;
          page-break-inside: avoid;
        }
        blockquote {
          border-left: 4px solid #ddd;
          padding-left: 10px;
          color: #666;
        }
        .page-break {
          page-break-after: always;
        }
        figure {
          margin: 1em 0;
          text-align: center;
          page-break-inside: avoid;
        }
        figcaption {
          font-style: italic;
          text-align: center;
          margin-top: 0.5em;
        }
      </style>
    </head>
    <body>
      ${htmlContent}
    </body>
    </html>
    `;
    
    // Write the HTML to a file
    const htmlFile = 'combined_research_paper.html';
    fs.writeFileSync(htmlFile, fullHtml);
    
    console.log('Generating PDF...');
    
    // Generate PDF using puppeteer with higher timeout
    const browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();
    
    // Set a longer timeout for page load
    await page.setDefaultNavigationTimeout(60000);
    
    // Load the HTML file (using file protocol)
    const htmlPath = path.resolve(htmlFile);
    await page.goto(`file://${htmlPath}`, { 
      waitUntil: 'networkidle0',
      timeout: 60000
    });
    
    // Wait a bit to ensure images are loaded
    await page.waitForTimeout(2000);
    
    // Configure PDF settings
    await page.pdf({
      path: 'Banana_Leaf_Disease_Research_Paper.pdf',
      format: 'A4',
      printBackground: true,
      margin: {
        top: '1cm',
        right: '1cm',
        bottom: '1cm',
        left: '1cm'
      },
      displayHeaderFooter: true,
      headerTemplate: '<div style="font-size: 8px; width: 100%; text-align: center; margin-top: 5px;">Banana Leaf Disease Research Paper</div>',
      footerTemplate: '<div style="font-size: 8px; width: 100%; text-align: center; margin-bottom: 5px;"><span class="pageNumber"></span> of <span class="totalPages"></span></div>'
    });
    
    await browser.close();
    
    console.log('PDF successfully generated: Banana_Leaf_Disease_Research_Paper.pdf');
    console.log('HTML file saved as: combined_research_paper.html');
    
  } catch (error) {
    console.error('Error generating PDF:', error);
  }
}

// Run the main function
generateResearchPaperPDF();