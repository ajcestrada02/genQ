$(document).ready(function(){
	// Setting the checkbox functions
	$('#btn_generate').attr('disabled', true);
	$('input[name="pageFrom"], input[name="pageTo"]').on('keyup',function() {
	    var pageFrom_name = $("#texta").val();
	    var pageFrom_value = $('input[name="pageFrom"]').val();
	    if(pageFrom_name != '' && pageFrom_value != '') {
	        $('input[name="btn_generate"]').attr('disabled' , false);
	    }else{
	        $('input[name="btn_generate"]').attr('disabled' , true);
	    }
	});

	// Changing the text content of input file
	$('input[type=file]').change(function(e){
		var fileName = event.target.files[0].name; // Getting the file name of chosen file
		if(fileName.length > 13){ // To limit the string of fileName
			fileName = fileName.substring(0,30) + '....';
		}
		$('input[type=file]').append('<style>.custom-file-control:after{content:"'+fileName+'";}</style>'); // Adding a style sheet in custom-file-control:after class
	});
	$('.carousel').carousel({
	  interval: 5000
	})
});

