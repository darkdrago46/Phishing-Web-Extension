
chrome.tabs.getSelected(null,function(tab) {
	var x = document.getElementById("loading-screen");
	var y = document.getElementById("result");
	x.style.display = "block";
	y.style.display = "none";
   	var tablink = tab.url;
	$("#url").text(tablink);
	$.post('http://localhost/Phishing-Detection/clientserver.php',{url:tablink},
	function(data)
	{
		x.style.display = "none";
		y.style.display = "block";
		$("#result").html(data);
		if(data=="NOT SECURE"){
			window.location.href ='alert.html';			
		}
	});
});