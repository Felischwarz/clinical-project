# From https://tex.stackexchange.com/questions/58963/latexmk-with-makeglossaries-and-auxdir-and-outdir#59098
add_cus_dep('glo', 'gls', 0, 'makeglossaries');
sub makeglossaries {
  my ($base_name, $path) = fileparse($_[0]);
  pushd $path;
  my $return = system "makeglossaries $base_name";
  popd;
  return $return;
}

# Use Biber for biblatex
$bibtex_use = 2;
$biber = '/Library/TeX/texbin/biber';

# Force biber instead of bibtex and respect build dir
$bibtex = '/Library/TeX/texbin/biber %O %B';

# Configure output and aux directories for all tools
$out_dir = 'build';
$aux_dir = 'build';

# Use explicit pdflatex path (let latexmk manage -outdir)
$pdflatex = '/Library/TeX/texbin/pdflatex %O -synctex=1 -interaction=nonstopmode -recorder %S';
