window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

    // BibTeX copy button
    $('.copy-bibtex').click(function() {
        var bibtex = $('#BibTeX pre code').text();
        navigator.clipboard.writeText(bibtex);
        var btn = $(this);
        btn.text('Copied!');
        setTimeout(function() { btn.text('Copy'); }, 2000);
    });

})
