$(document).ready(function() {
	$('div[data-toggle]').click(function() {
		$('#' + $(this).attr('data-toggle')).slideToggle('fast');
	});
});
