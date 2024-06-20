% Official PHreport LaTeX Document Class
% Written by Karl Welzel
% Last updated on 21 January 2024
% This document is published under a CC0 1.0 Universal License
% To view a copy of this license, visit http://creativecommons.org/publicdomain/zero/1.0

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{PHreportarticle}[PHreport Article Class]

% Provides macros for setting keys and declaring and setting class or package options
\RequirePackage{xkeyval}

\define@choicekey{PHreportarticle.cls}{area}[\PHreportarea\PHreportareanumber]{computerscience,socialsciences,humanities,mpls,bookreview}{}
\define@boolkey{PHreportarticle.cls}[]{journal}[true]{}
\define@boolkey{PHreportarticle.cls}[]{printing}[true]{}
\define@boolkey{PHreportarticle.cls}[]{review}[true]{}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\ProcessOptionsX\relax

% Unlike with \documentclass the options passed to \LoadClass are not automatically passed to all packages
% so we need to use \PassOptionsToPackage before the typearea package is loaded (within scrartcl)
\PassOptionsToPackage{paper=a4}{typearea}

\ifjournal
    \LoadClass[fontsize=11pt,twoside=true]{scrbook}
    \addtokomafont{partentry}{\scshape}
    \addtokomafont{part}{\scshape}
\else
    \ifreview
        \LoadClass[fontsize=11pt,twoside=false]{scrartcl}
    \else
        \LoadClass[fontsize=11pt,twoside=true]{scrartcl}
    \fi
\fi

%______________________________________________________________________________________
% Universal Packages and Settings
%______________________________________________________________________________________
\RequirePackage[utf8]{inputenc}
\RequirePackage[british]{babel}

\ifreview
    % Provides line numbers for the document to help with reviewing
    \usepackage[pagewise]{lineno}
    \AtBeginDocument{\linenumbers}
\fi

% Provides advanced facilities for inline and display quotations.
\RequirePackage{csquotes}

% The package allows the to user select any font size even those sizes that are not listed in the .fd file. If such a size is requested, LATEX will search for and select the nearest listed size; anyfontsize will then scale the font to the size actually requested.
\RequirePackage{anyfontsize}

% Provides typeface used throughout the document
\RequirePackage[osf]{Alegreya,AlegreyaSans}

% The package makes prettier tables
\RequirePackage{booktabs}

% Adds infix expressions to perform arithmetic on the arguments of the LATEX commands \setcounter, \addtocounter, \setlength, and \addtolength. Since many packages start their arithmetic by storing an argument in a register, the package has wide applicability.
\RequirePackage{calc}

% This package provides user control over the layout of the three basic list environments: enumerate, itemize and description.
\RequirePackage[shortlabels]{enumitem}

% The package allows the user to select font encodings, and for each encoding provides an interface to ‘font-encoding-specific’ commands for each font.
\RequirePackage[T1]{fontenc}

% EB Garamond is a revival by Georg Duffner of the 16th century fonts designed by Claude Garamond.
% \RequirePackage{ebgaramond}

% Standard package for inclusion of graphics in LaTeX
\RequirePackage{graphicx}

% The hyphenat package can be used to either disable hyphenation throughout a document or to enable automatic hyphenation within words that include analphabetic characters. It also provides for hyphenatable monospaced fonts.
% \RequirePackage{hyphenat}

% Generates filler text for display purposes
\RequirePackage{lipsum}

% Character protrusion and font expansion, furthermore the adjustment of interword spacing and additional kerning, as well as hyphenatable letterspacing (tracking) and the possibility to disable all or selected ligatures.
\RequirePackage[tracking=true,letterspace=50]{microtype}

% Allows for multiple columns in a single text column
% \RequirePackage{multicol}

% The package defines new commands \Centering, \RaggedLeft, and \RaggedRight and new environments Center, FlushLeft, and FlushRight, which set ragged text and are easily configurable
\RequirePackage{ragged2e}

% Provides support for setting the spacing between lines in a document. Package options include singlespacing, onehalfspacing, and doublespacing. Alternatively the spacing can be changed as required with the \singlespacing, \onehalfspacing, and \doublespacing commands.
% \RequirePackage{setspace}

% Driver-independent access to several kinds of color tints, shades, tones, and mixes of arbitrary colors. It allows a user to select a document-wide target color model and offers complete tools for conversion between eight color models.
\RequirePackage{xcolor}
\definecolor[named]{PHreportyellow}{HTML}{ffa92e}
\colorlet[named]{PHreportlinkcolor}{green!50!black}

% Provides \ifthenelse, which can be used in a wide array of tests.
\RequirePackage{ifthen}

% Provides proper spacing for \caption inside the table environment
% Make sure to always use \caption before \begin{tabular} so that the caption appears at the top!
\KOMAoptions{captions=tableheading}

% Increases space between table rows to increase legibitily
\renewcommand{\arraystretch}{1.4}

% Numbering within Document
\counterwithin{figure}{section}
\counterwithin{table}{section}

% Fix hyphenation for quite a lot of words, I don't know what's going on
\hyphenation{
    be-yond
    cog-ni-tive
    sen-si-tive
    in-sen-si-tive
    ob-jec-tive
    vari-abil-i-ty
    peri-ven-tric-u-lar
    sta-tis-ti-cal
    as-so-ci-at-ed
    in-di-vid-u-als
    seg-men-ta-tion
    com-par-isons
    phono-log-i-cal
    mul-ti-ple
    ob-served
    in-clud-ing
    analy-sis
    at-ro-phy
    as-sess-ments
    func-tion
    ed-u-ca-tion
    ef-fects
    com-par-isons
    as-sess
    com-bined
    car-dio-vas-cu-lar
    pro-cess-ing
    non-va-s-cu-lar
    se-lec-tion
    cog-ni-tive-ly
    sig-nif-i-cant
    neu-ropsy-cho-log-i-cal
    rep-re-sent
    sig-nif-i-cant-ly
    ex-ec-u-tive
    ed-u-ca-tion
    sub-cor-ti-cal
    in-de-pen-dent-ly
    in-di-cat-ing
    hip-pocam-pal
    seg-men-ta-tion
    for-nices
    in-di-vid-ual
    symp-tom
    there-fore
    re-duced
    in-tro-duces
    dis-ease
    sever-i-ty
    es-ti-mates
    se-mi-quan-ti-ta-tive
    pro-gres-sion
    pos-si-ble
    ob-jec-tive
    vari-abil-i-ty
    sub-cor-ti-cal
    gen-er-alised
    in-for-ma-tion
    ob-served
    in-volved
    in-farcts
    par-tic-i-pate
    neu-ropsy-cho-log-i-cal
}

\ifjournal
    % Skip loading area specific packages and bibliography styles
\else
    \ifx\PHreportarea\undefined
        \ClassError{PHreportarticle}{You must provide an `area=...` document class option}{}
    \fi
    
    %______________________________________________________________________________________
    % Referencing
    %______________________________________________________________________________________
    \ifcase\PHreportareanumber\relax
        % Social Sciences
        \RequirePackage[backend=biber,style=apa]{biblatex}
    \or
        % Humanities
        \RequirePackage[backend=biber,style=mla]{biblatex}
        \DeclareLanguageMapping{british}{american-apa}  % There is no British MLA, fall back to American version
    
        % Adapted from \fullcite in /usr/share/texlive/texmf-dist/tex/latex/biblatex-mla/mla.cbx and .../mla-strict.bbx because the MLA style does not provide a proper \fullcite command.
        \DeclareCiteCommand{\fullcite}
            {\usebibmacro{mla:prenote:see}}%
            {\usebibmacro{mla:container:top}%
             \printunit{\addperiod\space}\newblock%
             \usebibmacro{mla:container:one}%
             \printunit{\addperiod\space}\newblock%
             \usebibmacro{mla:container:two}}
            {\multicitedelim}%
            {\usebibmacro{postnote}}
    \or
        % MPLS
        \RequirePackage[backend=biber,style=numeric,maxbibnames=5]{biblatex}
    \or
        % Book review
        \RequirePackage[backend=biber,style=authoryear]{biblatex}
    \fi

    \renewcommand{\bibfont}{\footnotesize}
    
    
    \ifcase\PHreportareanumber\relax
        %______________________________________________________________________________________
        %Social Science Packages
        %______________________________________________________________________________________
    \or
        %______________________________________________________________________________________
        %Humanities Packages
        %______________________________________________________________________________________
    \or
        %______________________________________________________________________________________
        % MPLS Packages and Environments
        %______________________________________________________________________________________
        
        \RequirePackage{mathtools}
        \RequirePackage{amsfonts}
        \RequirePackage{amsthm}
    
        \AfterPackage{cleveref}{
            % This makes sure cleveref knows about the names of these environments
            \theoremstyle{plain}
            \newtheorem{theorem}{Theorem}[section]
            \newtheorem{proposition}[theorem]{Proposition}
            \newtheorem{lemma}[theorem]{Lemma}
            \newtheorem{corollary}[theorem]{Corollary}

            \theoremstyle{definition}
            \newtheorem{definition}[theorem]{Definition}
            
            \theoremstyle{remark}
            \newtheorem{remark}[theorem]{Remark}
            \newtheorem*{example}{Example}
            \newtheorem*{fact}{Fact}
        }
    \or
        %______________________________________________________________________________________
        % Book Review Packages
        %______________________________________________________________________________________
    \fi
\fi    

%______________________________________________________________________________________
% Section style
%______________________________________________________________________________________
\setkomafont{disposition}{\bfseries}
\setkomafont{section}{\fontsize{17}{17}\selectfont}
\setkomafont{subsection}{\fontsize{14}{14}\selectfont}
\setkomafont{subsubsection}{\fontsize{12}{12}\selectfont}


%______________________________________________________________________________________
% Links and references
%______________________________________________________________________________________
% Import these packages last
\RequirePackage[
    colorlinks=true,
    linkcolor=PHreportlinkcolor,
    citecolor=PHreportlinkcolor,
    filecolor=PHreportlinkcolor,
    urlcolor=blue,
]{hyperref}
\urlstyle{same}
\RequirePackage[capitalise,nameinlink]{cleveref}
\crefname{figure}{Figure}{Figures}  % Do not use abbreviations for referencing figures (no "fig. 3")

%______________________________________________________________________________________
% Custom title
%______________________________________________________________________________________

% Define keys for the \PHreportinfo command using xkeyval
\define@key{PHreportinfo}{title}{\hypersetup{pdftitle={#1}}}
\define@key{PHreportinfo}{author}{\hypersetup{pdfauthor={#1}}}
\define@cmdkeys{PHreportinfo}{
    shorttitle,
    subtitle,
    shortauthor,
    authorline,
    affiliation,
    actor,
    function,
    typeofdocument,
    publisheddate,
}

% Save the key arguments globaly as \XKV@KV@PHreportinfo@<key>@value where <key> is the name of the key
\savekeys{PHreportinfo}{
    \global{title},
    \global{shorttitle},
    \global{subtitle},
    \global{author},
    \global{shortauthor},
    \global{authorline},
    \global{affiliation},
    \global{actor},
    \global{function},
    \global{typeofdocument},
    \global{publisheddate},
}

% TODO: Make short title default to title if not given
\presetkeys{PHreportinfo}{subtitle={}}{}

\newcommand{\PHreportinfo}[1]{
    \setkeys{PHreportinfo}{#1}
}

\newlist{affiliations}{enumerate}{1}
\setlist[affiliations]{
    itemsep=0pt,
    topsep=0pt,
    label=\arabic*.,
    labelindent=0pt, 
    labelsep=.2em,
    leftmargin=*,
    align=left
} 

\ifjournal
    % Don't overwrite \maketitle, we use custom title page anyway
\else
    \ifthenelse{\equal{\PHreportarea}{bookreview}}
    {
        \renewcommand{\maketitle}{
            {
                % Put this in its own block, so the font change only affects the title
                \AlegreyaSans
                \setlength{\parindent}{0pt}
                {\AlegreyaSansMedium\fontsize{28}{28}\selectfont{}\XKV@KV@PHreportinfo@title@value\par}
            }
        }
    }{
        \renewcommand{\maketitle}{
            {
                % Put this in its own block, so the font change only affects the title
                \AlegreyaSans
                \setlength{\parindent}{0pt}
                {\AlegreyaSansMedium\fontsize{28}{28}\selectfont{}\XKV@KV@PHreportinfo@title@value\par}
                \ifthenelse{\not\equal{\XKV@KV@PHreportinfo@subtitle@value}{}}
                    {
                        % If there is a subtitle, print it
                        \vspace{.75em}
                        {\AlegreyaSansMedium\fontsize{20}{20}\selectfont{}\XKV@KV@PHreportinfo@subtitle@value\par}
                    }{
                        % otherwise, skip this line
                    }
                \vspace{1em}
                {\fontsize{16}{20}\selectfont{}\XKV@KV@PHreportinfo@authorline@value\par}
                \vspace{0.25em}
                {
                    \footnotesize
                    \begin{affiliations}\XKV@KV@PHreportinfo@affiliation@value\end{affiliations}
                }
            }
        }
    }
\fi


\newenvironment{PHreportabstractandkeywords}
    {
        \vspace{1.25em}
        \noindent
        \begin{minipage}[b]{.68\textwidth}
            \AlegreyaSans\small
            \rule{\textwidth}{0.4pt}
    }
    {
            \\[-.5em]
            \rule{\textwidth}{0.4pt}
        \end{minipage}\hfill
        \begin{minipage}[b]{.28\textwidth}
            \AlegreyaSans\small
            \setlength{\parskip}{.5em}
            \textbf{\XKV@KV@PHreportinfo@function@value:}\\
            \XKV@KV@PHreportinfo@actor@value\\[1cm]
            \textbf{Date de publication:} \XKV@KV@PHreportinfo@publisheddate@value

            \ifreview
                % Do not require authors to supply the "thispaper" citation
            \else
                \begin{refsection}
                    \textbf{Cite as:} \fullcite{thispaper}
                \end{refsection}
            \fi
        \end{minipage}
    }

\newenvironment{PHreportkeywords}[1][Keywords]
    {\\[1em]\noindent\RaggedRight\textbf{#1:} }
    {}

% Useful shorthand to make emails clickable
\newcommand{\email}[1]{\protect\href{mailto:#1}{#1}}

%______________________________________________________________________________________
% Book Reviews
%______________________________________________________________________________________

\newenvironment{PHreportbookreview}[3][nobibliography]
    {
        \provideboolean{PHreportprintbibliography}
        \ifthenelse{\not\equal{#1}{nobibliography}}
            {\setboolean{PHreportprintbibliography}{true}}
            {\setboolean{PHreportprintbibliography}{false}}
        
        % refsection makes sure the bibliography entries do not spill over
        \begin{refsection}[#2,]  % #2 = bibliography filename, e.g. bibliography.bib

        % Filter out all entries except for the book to be reviewed
        \defbibcheck{bookreviewkey}{
            \iffieldequalstr{entrykey}{#3} % #3 = entry key for the relevant BibTeX entry
                {}
                {\skipentry}
        }

        % Filter out the book to be reviewed
        \defbibcheck{notbookreviewkey}{
            \iffieldequalstr{entrykey}{#3}
                {\skipentry}
                {}
        }

        \nocite{#3}
        {
            % Put this in its own block, so it only affects this bibliography
            \renewcommand{\bibfont}{}
            \setlength{\bibhang}{0pt}
            \AtNextBibliography{\AlegreyaSans\bfseries}
            % Print only the entry for the book to be reviewed
            \printbibliography[heading=none,check=bookreviewkey]        
        }

        \@afterindentfalse\@afterheading
    }
    {
        % Print all but the entry for the book to be reviewed, if wanted
        \ifthenelse{\boolean{PHreportprintbibliography}}
            {\printbibliography[check=notbookreviewkey]}
            {}
        
        \end{refsection}
    }

\newcommand{\PHreportbookreviewauthor}[3][]{
    \addvspace{1.5em}
    \begin{samepage} % Make sure no page break happens here
        % Put this in its own block, so the alignment only affects these lines
        \RaggedLeft\AlegreyaSans
        \ifthenelse{\not\equal{#1}{}}{
            \includegraphics[height=3cm]{#1}\par
        }{
            % otherwise skip
        }
        \nopagebreak{}{\fontsize{16}{20}\textsc{#2}}\par   % #2 = Name of author
        \nopagebreak{}{#3}\par                             % #3 = Place and email
    \end{samepage}
}

%______________________________________________________________________________________
% Custom Headers
%______________________________________________________________________________________

\RequirePackage{scrlayer-scrpage}

\newpairofpagestyles[scrheadings]{firstpageheadings}{}

\ifjournal
    \newcommand{\PHreportheaders}{
        \KOMAoptions{
            headinclude=false,
            footinclude=false,
        }
        \setkomafont{pageheadfoot}{\AlegreyaSans}

        \pagestyle{scrheadings}
        \ihead{}
        \chead{}
        \ohead{}
        \if@twoside
           \cfoot{}
           \ofoot{\thepage}
        \else
           \cfoot{\thepage}
           \ofoot{}
        \fi

        \renewcommand*{\partpagestyle}{empty}
}
\else
    \newcommand{\PHreportheaders}[1][firstpage]{
        \KOMAoptions{
            headinclude=false,
            footinclude=false,
            headsepline=0.5pt,
        }
        \setkomafont{pageheadfoot}{\AlegreyaSans\color{gray}}
        \setkomafont{headsepline}{\color{gray}}
        \setkomafont{pagefoot}{\normalcolor}
        
        \pagestyle{firstpageheadings}
        \ihead{\XKV@KV@PHreportinfo@typeofdocument@value}  % inner heading
        \ifreview
           \ohead{REVIEW COPY}             % outer heading
        \else
           \ohead{\XKV@KV@PHreportinfo@publisheddate@value}
        \fi
        \if@twoside
           \cfoot{}
           \ofoot{\thepage}
        \else
           \cfoot{\thepage}
           \ofoot{}
        \fi
        
        \pagestyle{scrheadings}
        \ihead{\XKV@KV@PHreportinfo@shortauthor@value}      % inner heading
        \ifthenelse{\equal{\PHreportarea}{bookreview}}      % outer heading
            {\ohead{\XKV@KV@PHreportinfo@publisheddate@value}}
            {\ohead{\XKV@KV@PHreportinfo@shorttitle@value}}
        \if@twoside
           \cfoot{}
           \ofoot{\thepage}
        \else
           \cfoot{\thepage}
           \ofoot{}
        \fi
    
        \ifthenelse{\equal{#1}{firstpage}}
            {
                \thispagestyle{firstpageheadings}
            }{
                % We are not on the first page, use regular headings
            }
    }
\fi

%______________________________________________________________________________________
% Margins
%______________________________________________________________________________________

\KOMAoptions{
    areasetadvanced=true,
}
% Distribute margins according to the "canons of page construction", see scrartcl documentation
% The result is similar to DIV=12. It produces ~2.5cm left and right margins, ~2.5cm at the top and ~5cm at the bottom for oneside
% Header and footer are counted as part of the margin
\ifprinting
    % Add binding correction (2cm)
    \areaset[2cm]{\paperwidth - 5cm}{\paperheight - 7.5cm}
\else
    \areaset{\paperwidth - 5cm}{\paperheight - 7.5cm}    
\fi

% \ifprinting
%     \usepackage[cross, noinfo, width=230truemm, height=317truemm, center, pdflatex]{crop}
% \fi